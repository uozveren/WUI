import streamlit as st
import subprocess
import os
import yaml
import threading
import time
import datetime
import csv
import pandas as pd
from io import StringIO

# Ek metrik/grafik kütüphaneleri
try:
    from evaluate import load as load_metric  # BLEU, ROUGE vb.
except ImportError:
    load_metric = None

try:
    import matplotlib.pyplot as plt
    import plotly.express as px
except ImportError:
    pass

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
except ImportError:
    pass

########################
# GLOBAL DEĞİŞKENLER
########################

training_process = None
tensorboard_process = None

# Deney kaydı CSV
EXPERIMENT_LOG_FILE = "experiment_logs.csv"

# Otomatik OOM'da batch düşürme
AUTO_RETRY_ON_OOM = True

########################
# 0) Kütüphane Kurma
########################

def install_dependencies(gpu_enabled=False):
    """
    GPU seçildiyse CUDA destekli paketler; CPU için normal paketler yüklenecek.
    Bu işlemler uygulama içinden yapılırsa, Streamlit bir sonraki yenilemede güncellenmiş olur.
    """
    if gpu_enabled:
        # CUDA destekli PyTorch, Unsloth, vb.
        dependencies = [
            "unsloth",
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",  # Örn. CUDA 11.8
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "tensorboard",
            "pyyaml",
            "evaluate",  # BLEU, ROUGE
            "scikit-learn",
            "matplotlib",
            "plotly"
        ]
    else:
        # CPU versiyon paketleri
        dependencies = [
            "unsloth",
            "torch torchvision torchaudio",
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "tensorboard",
            "pyyaml",
            "evaluate",
            "scikit-learn",
            "matplotlib",
            "plotly"
        ]

    install_log = ""
    for pkg in dependencies:
        cmd = ["pip", "install"] + pkg.split()
        install_log += run_subprocess(cmd, realtime=False)
    return install_log

########################
# Yardımcı Fonksiyonlar
########################

def run_subprocess(cmd_list, realtime=True):
    """
    Komutu alt süreçte (subprocess) çalıştır.
    realtime=True: çıktı anlık Streamlit'te gösterilir.
    """
    global training_process

    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    training_process = process

    output_buffer = []
    if realtime:
        log_placeholder = st.empty()
    else:
        log_placeholder = None

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            if realtime and log_placeholder:
                log_placeholder.text(line.rstrip())
            output_buffer.append(line)

    rc = process.poll()
    training_process = None
    return "".join(output_buffer)

def stop_training():
    """
    Çalışan eğitim sürecini durdurur.
    """
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        st.warning("Eğitim işlemi durduruldu!")
        training_process = None
    else:
        st.info("Aktif bir eğitim süreci bulunamadı.")

def load_existing_config(config_path):
    """
    config.yaml dosyasını sözlük olarak okur.
    """
    if not os.path.exists(config_path):
        st.error(f"{config_path} bulunamadı.")
        return {}
    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            st.error(f"YAML okuma hatası: {e}")
            return {}

def write_config_file(
    base_model,
    train_data_path,
    test_data_path,
    output_dir,
    max_steps,
    train_batch_size,
    grad_acc_steps,
    learning_rate,
    scheduler,
    resume_from_checkpoint=None,
    use_qlora=False
):
    """
    config.yaml oluştur/güncelle.
    """
    config_dict = {
        "base_model": base_model,
        "data": {
            "path": train_data_path,
            "type": "alpaca"
        },
        "evaluation": {
            "test_data_path": test_data_path,
            "metrics": ["accuracy", "f1", "bleu", "rouge"]  # Örnek ek metrik
        },
        "training": {
            "output_dir": output_dir,
            "max_steps": max_steps,
            "per_device_train_batch_size": train_batch_size,
            "gradient_accumulation_steps": grad_acc_steps,
            "save_steps": 50,
            "logging_steps": 10,
            "fp16": True,
            "lr": float(learning_rate),
            "warmup_steps": 10,
            "lr_scheduler_type": scheduler,
            "gradient_checkpointing": True
        },
        "use_qlora": use_qlora
    }
    if resume_from_checkpoint:
        config_dict["training"]["resume_from_checkpoint"] = resume_from_checkpoint

    with open("config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    st.success("config.yaml oluşturuldu/güncellendi!")

########################
# Deney Kayıtları
########################
def init_experiment_logs():
    if not os.path.exists(EXPERIMENT_LOG_FILE):
        with open(EXPERIMENT_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "params", "metrics", "notes"])

def log_experiment(params_str, metrics_str, notes=""):
    init_experiment_logs()
    timestamp = datetime.datetime.now().isoformat()
    with open(EXPERIMENT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, params_str, metrics_str, notes])

def load_experiment_logs():
    if not os.path.exists(EXPERIMENT_LOG_FILE):
        return pd.DataFrame(columns=["timestamp", "params", "metrics", "notes"])
    return pd.read_csv(EXPERIMENT_LOG_FILE)

########################
# TensorBoard Yönetimi
########################
def start_tensorboard(logdir="output"):
    global tensorboard_process

    def run_tb():
        cmd = ["tensorboard", "--logdir", logdir, "--port", "6006"]
        tensorboard_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tensorboard_process.wait()

    tb_thread = threading.Thread(target=run_tb, daemon=True)
    tb_thread.start()
    st.info("TensorBoard başlatıldı. Bu sayfanın alt kısmındaki iFrame veya http://localhost:6006 üzerinden erişebilirsiniz.")

def stop_tensorboard():
    global tensorboard_process
    if tensorboard_process and tensorboard_process.poll() is None:
        tensorboard_process.terminate()
        st.warning("TensorBoard durduruldu!")
        tensorboard_process = None
    else:
        st.info("Aktif bir TensorBoard süreci bulunamadı.")

########################
# GPU OOM Yakalama
########################
def attempt_training_with_auto_oom_fix(initial_cmd, config_params):
    max_retries = 3
    current_retry = 0
    logs_all = ""

    while current_retry < max_retries:
        st.info(f"Eğitim denemesi {current_retry+1}/{max_retries} başlatılıyor...")
        logs = run_subprocess(initial_cmd, realtime=True)
        logs_all += logs
        if "CUDA out of memory" in logs:
            st.warning("GPU bellek hatası (OOM) algılandı! Batch size otomatik düşürülüyor.")
            config_params["train_batch_size"] = max(1, config_params["train_batch_size"] // 2)
            st.info(f"Yeni batch_size={config_params['train_batch_size']} ile tekrar denenecek.")

            write_config_file(
                base_model=config_params["base_model"],
                train_data_path=config_params["train_data_path"],
                test_data_path=config_params["test_data_path"],
                output_dir=config_params["output_dir"],
                max_steps=config_params["max_steps"],
                train_batch_size=config_params["train_batch_size"],
                grad_acc_steps=config_params["grad_acc_steps"],
                learning_rate=config_params["learning_rate"],
                scheduler=config_params["scheduler"],
                resume_from_checkpoint=config_params["resume_from_checkpoint"],
                use_qlora=config_params["use_qlora"]
            )
            initial_cmd = ["python3", "-m", "unsloth.cli", "--config", "config.yaml"]
            current_retry += 1
        else:
            break

    return logs_all

########################
# Ek Metrik Hesaplama (ROUGE, BLEU vb.) + Confusion Matrix / ROC
########################
def compute_extra_metrics(predictions, references):
    """
    Örnek: BLEU, ROUGE gibi metrikler için 'evaluate' kütüphanesini kullan.
    predictions, references: List[str]
    """
    results = {}
    if load_metric is None:
        st.warning("Ek metrikler (ROUGE, BLEU vb.) için 'evaluate' kütüphanesi bulunamadı.")
        return results

    # BLEU
    try:
        bleu_metric = load_metric("bleu")
        ref_wrapped = [[r.split()] for r in references]
        pred_wrapped = [p.split() for p in predictions]
        bleu_res = bleu_metric.compute(predictions=pred_wrapped, references=ref_wrapped)
        results["BLEU"] = bleu_res["bleu"]
    except Exception as e:
        st.error(f"BLEU hesaplama hatası: {e}")

    # ROUGE
    try:
        rouge_metric = load_metric("rouge")
        rouge_res = rouge_metric.compute(predictions=predictions, references=references)
        results["ROUGE-1"] = rouge_res["rouge1"]
        results["ROUGE-2"] = rouge_res["rouge2"]
        results["ROUGE-L"] = rouge_res["rougeL"]
    except Exception as e:
        st.error(f"ROUGE hesaplama hatası: {e}")

    return results


def display_confusion_matrix(labels, preds):
    """
    Confusion Matrix ve ROC grafiklerini gösterir.
    labels, preds: List[int] veya benzer
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm)
    fig_cm, ax_cm = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
    st.pyplot(fig_cm)

    # ROC Eğrisi
    # Basit örnek: 2 sınıf (pozitif/negatif) varsayıyor.
    # preds bir olasılık yerine etiket. Normalde preds=0/1 vs.
    # Demo olarak y_score = preds
    try:
        fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC={roc_auc:.2f})')
        ax_roc.plot([0,1],[0,1], color='gray', linestyle='--')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)
    except Exception as e:
        st.warning(f"ROC eğrisi çizilirken hata: {e}")


########################
# Streamlit Ana Uygulama
########################

def main():
    st.title("Unsloth Fine-Tuning Web UI (Çok Kapsamlı)")

    st.markdown("""
    Bu uygulama, **CPU/GPU seçimi**, **TensorBoard iFrame** gösterimi, 
    **ek metrik (ROUGE, BLEU, Confusion Matrix, ROC)** ve 
    **GPU bellek hatasında otomatik batch size düşürme** dahil birçok ileri özelliği içerir.
    """)

    ########################
    # Kullanıcıdan CPU/GPU Seçimi
    ########################
    st.subheader("CPU veya GPU Seçimi")
    compute_choice = st.radio("Eğitim nerede yapılacak?", ["CPU", "GPU"], index=0)
    if compute_choice == "GPU":
        st.info("GPU seçildi. CUDA destekli kütüphaneler kurulması gerekebilir.")
    else:
        st.info("CPU üzerinden eğitim yapılacak.")

    if st.button("Kütüphaneleri Kur (CPU/GPU)"):
        st.info("Kurulum başlıyor, bu işlem biraz zaman alabilir...")
        logs = install_dependencies(gpu_enabled=(compute_choice == "GPU"))
        st.text(logs)
        st.success("Paket kurulum işlemi tamamlandı (Uygulama yeniden başlatılabilir).")

    st.write("---")

    ########################
    # Bölüm 1: GPU Bilgisi
    ########################
    st.subheader("1) GPU Bilgisi (nvidia-smi)")
    if st.button("GPU Bilgisini Göster"):
        try:
            gpu_info = subprocess.check_output(["nvidia-smi"], text=True)
            st.code(gpu_info)
        except FileNotFoundError:
            st.error("nvidia-smi komutu bulunamadı veya GPU desteği yok.")

    st.write("---")

    ########################
    # Bölüm 2: Veri Seti Yükleme
    ########################
    st.subheader("2) Veri Setlerini Yükleme (JSONL)")

    up_train = st.file_uploader("Train Dataset (train.jsonl)", type=["jsonl"])
    if up_train is not None:
        os.makedirs("datasets", exist_ok=True)
        train_path = os.path.join("datasets", "train.jsonl")
        with open(train_path, "wb") as f:
            f.write(up_train.getbuffer())
        st.success(f"Train veri seti: {train_path}")

    up_test = st.file_uploader("Test Dataset (test.jsonl)", type=["jsonl"])
    if up_test is not None:
        os.makedirs("datasets", exist_ok=True)
        test_path = os.path.join("datasets", "test.jsonl")
        with open(test_path, "wb") as f:
            f.write(up_test.getbuffer())
        st.success(f"Test veri seti: {test_path}")

    st.write("---")

    ########################
    # Bölüm 3: Config Dosyası
    ########################
    st.subheader("3) Config Dosyası (YAML)")

    # Mevcut config.yaml Yükleme
    st.markdown("**Mevcut config.yaml Yükleme:**")
    existing_cfg_path = st.text_input("Mevcut config.yaml", "existing_config.yaml")
    if st.button("Config Yükle"):
        loaded_config = load_existing_config(existing_cfg_path)
        if loaded_config:
            st.json(loaded_config)

    st.markdown("---")

    # Yeni config.yaml
    st.markdown("**Yeni config.yaml oluşturun:**")
    base_model = st.text_input("Base Model", "meta-llama/Llama-2-7b-hf")
    train_data = st.text_input("Train Data Path", "datasets/train.jsonl")
    test_data = st.text_input("Test Data Path", "datasets/test.jsonl")
    output_dir = st.text_input("Çıktı Klasörü", "output")

    c1, c2 = st.columns(2)
    with c1:
        max_steps = st.number_input("Max Steps", min_value=1, value=100, step=10)
        grad_acc_steps = st.number_input("Gradient Acc Steps", min_value=1, value=8)
    with c2:
        train_batch_size = st.number_input("Train Batch Size", min_value=1, value=2)
        learning_rate = st.text_input("Learning Rate", "2e-5")

    scheduler = st.selectbox("LR Scheduler", ["linear", "cosine", "constant"], index=1)
    resume_cp = st.text_input("Resume from Checkpoint (opsiyonel)", "")
    use_qlora = st.checkbox("QLoRA Kullan?")

    if st.button("Config Oluştur/Güncelle"):
        write_config_file(
            base_model=base_model,
            train_data_path=train_data,
            test_data_path=test_data,
            output_dir=output_dir,
            max_steps=max_steps,
            train_batch_size=train_batch_size,
            grad_acc_steps=grad_acc_steps,
            learning_rate=learning_rate,
            scheduler=scheduler,
            resume_from_checkpoint=resume_cp,
            use_qlora=use_qlora
        )

    st.write("---")

    ########################
    # Bölüm 4: Eğitim Başlatma
    ########################
    st.subheader("4) Eğitim (Fine-Tuning)")

    train_config_dict = {
        "base_model": base_model,
        "train_data_path": train_data,
        "test_data_path": test_data,
        "output_dir": output_dir,
        "max_steps": max_steps,
        "train_batch_size": train_batch_size,
        "grad_acc_steps": grad_acc_steps,
        "learning_rate": learning_rate,
        "scheduler": scheduler,
        "resume_from_checkpoint": resume_cp,
        "use_qlora": use_qlora
    }

    if st.button("Eğitimi Başlat"):
        if not os.path.exists("config.yaml"):
            st.error("config.yaml bulunamadı! Lütfen önce oluşturun veya yükleyin.")
        else:
            st.info("Eğitim başlatılıyor...")
            cmd = ["python3", "-m", "unsloth.cli", "--config", "config.yaml"]
            if AUTO_RETRY_ON_OOM and compute_choice == "GPU":
                logs = attempt_training_with_auto_oom_fix(cmd, train_config_dict)
            else:
                logs = run_subprocess(cmd, realtime=True)
            st.text_area("Eğitim Log", logs, height=200)

    if st.button("Eğitimi Durdur"):
        stop_training()

    st.write("---")

    ########################
    # Bölüm 5: Değerlendirme, Ek Metrikler, Confusion Matrix, ROC
    ########################
    st.subheader("5) Değerlendirme & Ek Metrikler")

    evaluate_button = st.button("Değerlendirme Yap")
    if evaluate_button:
        if not os.path.exists(output_dir):
            st.error(f"{output_dir} klasörü yok, önce eğitimi tamamlayın.")
        else:
            st.info("Değerlendirme başlıyor...")
            eval_cmd = [
                "python3",
                "-m",
                "unsloth.evaluate",
                "--model_dir", output_dir,
                "--test_data", test_data
            ]
            eval_log = run_subprocess(eval_cmd, realtime=True)
            st.text_area("Değerlendirme Log", eval_log, height=200)

            # Burada ek metrikler, confusion matrix, ROC gibi görseller gösterelim.
            st.markdown("### Ek Metrikler ve Grafikler")

            # Örnek dummy predictions & references (normalde model çıktısından/parsed logs'tan elde edilir)
            predictions = ["hello world", "this is a test"]
            references  = ["hello new world", "this is test"]

            extra_metrics = compute_extra_metrics(predictions, references)
            st.write("ROUGE/BLEU Sonuçları:", extra_metrics)

            # Confusion Matrix / ROC için dummy label/preds
            # Normalde unsloth.evaluate çıktısını parse edip (csv vs.) label/pred elde etmelisiniz.
            dummy_labels = [0, 1, 1, 0, 1, 1]
            dummy_preds  = [0, 1, 0, 0, 1, 1]

            display_confusion_matrix(dummy_labels, dummy_preds)

            # Geçmiş deney kaydına yazabiliriz
            param_info = f"model={base_model}, steps={max_steps}, batch_size={train_batch_size}"
            metrics_info = str(extra_metrics)
            log_experiment(param_info, metrics_info, notes="Ek metrik + ConfMat/ROC deneme")

            st.success("Değerlendirme tamamlandı, ek metrik ve grafikleri görüntülediniz.")

    st.write("---")

    ########################
    # Bölüm 6: Sonuçları CSV
    ########################
    st.subheader("6) Sonuçları CSV Olarak Kaydet & İndir")

    csv_name = st.text_input("CSV Dosya Adı", "results.csv")
    if st.button("CSV'ye Kaydet"):
        if not os.path.exists(output_dir):
            st.error(f"{output_dir} klasörü yok.")
        else:
            st.info("CSV'ye kaydediliyor...")
            csv_cmd = [
                "python3", 
                "-m", 
                "unsloth.evaluate",
                "--model_dir", 
                output_dir,
                "--test_data", 
                test_data,
                "--output_file", 
                csv_name
            ]
            csv_log = run_subprocess(csv_cmd, realtime=True)
            st.text_area("CSV Log", csv_log, height=100)

    if os.path.exists(csv_name):
        with open(csv_name, "rb") as f:
            st.download_button(
                label="CSV Dosyasını İndir",
                data=f,
                file_name=csv_name,
                mime="text/csv"
            )

    st.write("---")

    ########################
    # Bölüm 7: TensorBoard + iFrame
    ########################
    st.subheader("7) TensorBoard")
    tb_dir = st.text_input("TensorBoard Log Dir", output_dir)
    col_tb1, col_tb2 = st.columns(2)
    if col_tb1.button("TensorBoard Başlat"):
        if not os.path.exists(tb_dir):
            st.error(f"{tb_dir} bulunamadı!")
        else:
            start_tensorboard(tb_dir)
    if col_tb2.button("TensorBoard Durdur"):
        stop_tensorboard()

    st.markdown("**TensorBoard Arayüzü (iFrame denemesi):**")
    # iFrame içinden 6006 portunu gösterelim. 
    # Not: Sunucuda iframe politika sorunları veya port sorunları olabilir.
    # "http://localhost:6006" -> 
    st.components.v1.iframe(src="http://localhost:6006", height=600, scrolling=True)

    st.write("---")

    ########################
    # Bölüm 8: Geçmiş Deney Kayıtları
    ########################
    st.subheader("8) Geçmiş Deney Kayıtları")
    df_logs = load_experiment_logs()
    st.dataframe(df_logs)

    if not df_logs.empty:
        st.write("Örnek Dinamik Grafik (ROUGE-L)")
        def parse_rougeL(m):
            try:
                if isinstance(m, str) and "ROUGE-L" in m:
                    val_str = m.split("ROUGE-L")[1].split(":")[1].split(",")[0]
                    return float(val_str)
            except:
                pass
            return None

        df_logs["rougeL_value"] = df_logs["metrics"].apply(parse_rougeL)
        st.line_chart(df_logs["rougeL_value"].dropna())

    st.info("Uygulama sonu: Tüm özellikler burada. Keyifli denemeler!")
    
if __name__ == "__main__":
    main()
