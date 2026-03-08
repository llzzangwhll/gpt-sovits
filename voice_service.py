"""
GPT-SoVITS 음성 학습 및 TTS 서비스
- 학습 UI: 음성 데이터로 모델 학습
- TTS 테스트 UI: 학습된 모델로 음성 합성 테스트
"""

import json
import os
import platform
import re
import shutil
import signal
import sys
import traceback
from io import BytesIO
from multiprocessing import cpu_count
from subprocess import Popen

import numpy as np
import psutil
import soundfile as sf
import yaml

now_dir = os.getcwd()
sys.path.insert(0, now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

os.environ["version"] = "v2"

import gradio as gr

from config import (
    GPU_INDEX,
    GPU_INFOS,
    IS_GPU,
    exp_root,
    infer_device,
    is_half,
    memset,
    python_exec,
    pretrained_gpt_name,
    pretrained_sovits_name,
    GPT_weight_root,
    GPT_weight_version2root,
    SoVITS_weight_root,
    SoVITS_weight_version2root,
    get_weights_names,
)

# ========== 설정 ==========
VERSION = os.environ.get("version", "v2")
SUPPORTED_VERSIONS = ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"]
DEFAULT_GPU = str(infer_device.index) if infer_device.type != "cpu" else "0"
SYSTEM = platform.system()

for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)

# ========== 프로세스 관리 ==========
running_processes = {}


def kill_process(pid, name=""):
    try:
        if SYSTEM == "Windows":
            Popen(f"taskkill /t /f /pid {pid}", shell=True,
                  stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))
        else:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    os.kill(child.pid, signal.SIGTERM)
                except OSError:
                    pass
            os.kill(parent.pid, signal.SIGTERM)
    except Exception:
        pass
    print(f"[{name}] 프로세스 종료됨")


def stop_process(key):
    if key in running_processes and running_processes[key] is not None:
        proc = running_processes[key]
        if isinstance(proc, list):
            for p in proc:
                kill_process(p.pid, key)
        else:
            kill_process(proc.pid, key)
        running_processes[key] = None
        return f"[{key}] 중지됨"
    return f"[{key}] 실행 중인 프로세스 없음"


# ========== 모델 가중치 스캔 ==========
def scan_weights():
    sovits_names, gpt_names = get_weights_names()
    return sovits_names, gpt_names


def get_version_defaults(ver):
    v3v4 = {"v3", "v4"}
    if IS_GPU:
        min_mem = min(memset) if memset else 4
        batch = int(min_mem // 2 if ver not in v3v4 else min_mem // 8)
        batch_s1 = int(min_mem // 2)
    else:
        total_mem = psutil.virtual_memory().total / (1024 ** 3)
        batch = batch_s1 = int(total_mem / 4)

    batch = max(1, batch)
    batch_s1 = max(1, batch_s1)

    if ver not in v3v4:
        sovits_epoch = 8
        save_every = 4
    else:
        sovits_epoch = 2
        save_every = 1

    return batch, batch_s1, sovits_epoch, save_every


# ========== 1. 오디오 슬라이싱 ==========
def run_slice(inp_path, out_path, threshold=-34, min_length=4000,
              min_interval=300, hop_size=10, max_sil_kept=500,
              max_amp=0.9, alpha=0.25):
    if not inp_path or not os.path.exists(inp_path):
        yield "오디오 경로가 존재하지 않습니다"
        return

    out_path = out_path or "output/slicer_opt"
    os.makedirs(out_path, exist_ok=True)

    n_parts = 1 if os.path.isfile(inp_path) else cpu_count()
    n_parts = min(n_parts, 4)

    procs = []
    for i in range(n_parts):
        cmd = (f'"{python_exec}" -s tools/slice_audio.py '
               f'"{inp_path}" "{out_path}" {threshold} {min_length} '
               f'{min_interval} {hop_size} {max_sil_kept} {max_amp} {alpha} '
               f'{i} {n_parts}')
        print(cmd)
        procs.append(Popen(cmd, shell=True))

    running_processes["slice"] = procs
    yield "오디오 슬라이싱 진행 중..."

    for p in procs:
        p.wait()
    running_processes["slice"] = None
    yield f"오디오 슬라이싱 완료! 출력: {out_path}"


# ========== 2. 음성 인식 (ASR) ==========
def run_asr(inp_dir, out_dir, model="Faster Whisper (多语종)",
            model_size="large-v3", lang="ko", precision="float16"):
    if not inp_dir or not os.path.exists(inp_dir):
        yield "입력 경로가 존재하지 않습니다"
        return

    out_dir = out_dir or "output/asr_opt"
    os.makedirs(out_dir, exist_ok=True)

    asr_scripts = {
        "达摩 ASR (中文)": "funasr_asr.py",
        "Faster Whisper (多语종)": "fasterwhisper_asr.py",
    }
    script = asr_scripts.get(model, "fasterwhisper_asr.py")

    cmd = (f'"{python_exec}" -s tools/asr/{script} '
           f'-i "{inp_dir}" -o "{out_dir}" '
           f'-s {model_size} -l {lang} -p {precision}')
    print(cmd)

    p = Popen(cmd, shell=True)
    running_processes["asr"] = p
    yield "음성 인식(ASR) 진행 중..."

    p.wait()
    running_processes["asr"] = None

    # 결과 파일 찾기
    base_name = os.path.basename(inp_dir)
    result_path = os.path.join(os.path.abspath(out_dir), f"{base_name}.list")
    if os.path.exists(result_path):
        yield f"ASR 완료! 결과 파일: {result_path}"
    else:
        yield f"ASR 완료! 출력 디렉토리: {out_dir}"


# ========== 3. 데이터셋 전처리 (1A+1B+1C 일괄) ==========
def run_dataset_prep(version, inp_text, inp_wav_dir, exp_name, gpu_number=DEFAULT_GPU):
    if not inp_text or not inp_wav_dir:
        yield "텍스트 리스트 파일과 오디오 디렉토리를 모두 지정하세요"
        return

    if not os.path.exists(inp_text):
        yield f"텍스트 파일이 존재하지 않습니다: {inp_text}"
        return

    exp_name = exp_name.strip()
    if not exp_name:
        yield "실험 이름을 입력하세요"
        return

    opt_dir = os.path.join(exp_root, exp_name)
    os.makedirs(opt_dir, exist_ok=True)

    bert_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    ssl_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
    pretrained_s2G = pretrained_sovits_name.get(version, "")

    config_file = ("GPT_SoVITS/configs/s2.json"
                   if version not in {"v2Pro", "v2ProPlus"}
                   else f"GPT_SoVITS/configs/s2{version}.json")

    # === Step 1A: 텍스트 분석 ===
    yield "진행: 1A - 텍스트 분석 중..."
    path_text = os.path.join(opt_dir, "2-name2text.txt")

    config = {
        "inp_text": inp_text,
        "inp_wav_dir": inp_wav_dir,
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "bert_pretrained_dir": bert_dir,
        "is_half": str(is_half),
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": gpu_number,
    }
    os.environ.update(config)

    cmd = f'"{python_exec}" -s GPT_SoVITS/prepare_datasets/1-get-text.py'
    print(cmd)
    p = Popen(cmd, shell=True)
    running_processes["prep"] = p
    p.wait()

    # 병합
    txt_path = os.path.join(opt_dir, "2-name2text-0.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf8") as f:
            opt = f.read().strip("\n").split("\n")
        os.remove(txt_path)
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")

    if not os.path.exists(path_text):
        yield "1A 텍스트 분석 실패"
        return

    yield "진행: 1A 완료, 1B - 음성 특성 추출 중..."

    # === Step 1B: HuBERT 특성 추출 ===
    config.update({
        "cnhubert_base_dir": ssl_dir,
        "sv_path": sv_path,
    })
    os.environ.update(config)

    cmd = f'"{python_exec}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
    print(cmd)
    p = Popen(cmd, shell=True)
    running_processes["prep"] = p
    p.wait()

    # Pro 버전: SV 추출
    if "Pro" in version:
        cmd = f'"{python_exec}" -s GPT_SoVITS/prepare_datasets/2-get-sv.py'
        print(cmd)
        p = Popen(cmd, shell=True)
        p.wait()

    yield "진행: 1A 완료, 1B 완료, 1C - 시맨틱 토큰 추출 중..."

    # === Step 1C: 시맨틱 토큰 추출 ===
    config.update({
        "pretrained_s2G": pretrained_s2G,
        "s2config_path": config_file,
    })
    os.environ.update(config)

    cmd = f'"{python_exec}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py'
    print(cmd)
    p = Popen(cmd, shell=True)
    running_processes["prep"] = p
    p.wait()

    # 병합
    semantic_path = os.path.join(opt_dir, "6-name2semantic.tsv")
    part_path = os.path.join(opt_dir, "6-name2semantic-0.tsv")
    if os.path.exists(part_path):
        opt_lines = ["item_name\tsemantic_audio"]
        with open(part_path, "r", encoding="utf8") as f:
            opt_lines += f.read().strip("\n").split("\n")
        os.remove(part_path)
        with open(semantic_path, "w", encoding="utf8") as f:
            f.write("\n".join(opt_lines) + "\n")

    running_processes["prep"] = None
    yield "데이터셋 전처리 완료! (1A + 1B + 1C)"


# ========== 4. SoVITS 학습 ==========
def run_sovits_train(version, exp_name, batch_size, total_epoch,
                     save_every_epoch, text_low_lr_rate=0.4,
                     if_save_latest=True, if_save_every_weights=True,
                     gpu_numbers=DEFAULT_GPU):
    exp_name = exp_name.strip()
    if not exp_name:
        yield "실험 이름을 입력하세요"
        return

    s2_dir = os.path.join(exp_root, exp_name)
    os.makedirs(os.path.join(s2_dir, f"logs_s2_{version}"), exist_ok=True)

    config_file = ("GPT_SoVITS/configs/s2.json"
                   if version not in {"v2Pro", "v2ProPlus"}
                   else f"GPT_SoVITS/configs/s2{version}.json")

    with open(config_file) as f:
        data = json.loads(f.read())

    pretrained_s2G = pretrained_sovits_name.get(version, "")
    pretrained_s2D = pretrained_s2G.replace("s2G", "s2D") if pretrained_s2G else ""

    if not is_half:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)

    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = gpu_numbers
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = 0
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
    data["save_weight_dir"] = SoVITS_weight_version2root[version]
    data["name"] = exp_name
    data["version"] = version

    tmp_dir = os.path.join(now_dir, "TEMP")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_config = os.path.join(tmp_dir, "tmp_s2.json")
    with open(tmp_config, "w") as f:
        f.write(json.dumps(data))

    if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
        cmd = f'"{python_exec}" -s GPT_SoVITS/s2_train.py --config "{tmp_config}"'
    else:
        cmd = f'"{python_exec}" -s GPT_SoVITS/s2_train_v3_lora.py --config "{tmp_config}"'

    print(cmd)
    yield "SoVITS 학습 시작..."

    p = Popen(cmd, shell=True)
    running_processes["sovits_train"] = p
    p.wait()
    running_processes["sovits_train"] = None

    yield "SoVITS 학습 완료!"


# ========== 5. GPT 학습 ==========
def run_gpt_train(version, exp_name, batch_size, total_epoch,
                  save_every_epoch, if_dpo=False,
                  if_save_latest=True, if_save_every_weights=True,
                  gpu_numbers=DEFAULT_GPU):
    exp_name = exp_name.strip()
    if not exp_name:
        yield "실험 이름을 입력하세요"
        return

    config_yaml = ("GPT_SoVITS/configs/s1longer.yaml"
                   if version == "v1"
                   else "GPT_SoVITS/configs/s1longer-v2.yaml")

    with open(config_yaml) as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)

    s1_dir = os.path.join(exp_root, exp_name)
    os.makedirs(os.path.join(s1_dir, "logs_s1"), exist_ok=True)

    pretrained_s1 = pretrained_gpt_name.get(version, "")

    if not is_half:
        data["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)

    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["pretrained_s1"] = pretrained_s1
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]
    data["train"]["exp_name"] = exp_name
    data["train_semantic_path"] = os.path.join(s1_dir, "6-name2semantic.tsv")
    data["train_phoneme_path"] = os.path.join(s1_dir, "2-name2text.txt")
    data["output_dir"] = os.path.join(s1_dir, f"logs_s1_{version}")

    os.environ["_CUDA_VISIBLE_DEVICES"] = gpu_numbers
    os.environ["hz"] = "25hz"

    tmp_dir = os.path.join(now_dir, "TEMP")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_config = os.path.join(tmp_dir, "tmp_s1.yaml")
    with open(tmp_config, "w") as f:
        f.write(yaml.dump(data, default_flow_style=False))

    cmd = f'"{python_exec}" -s GPT_SoVITS/s1_train.py --config_file "{tmp_config}"'
    print(cmd)
    yield "GPT 학습 시작..."

    p = Popen(cmd, shell=True)
    running_processes["gpt_train"] = p
    p.wait()
    running_processes["gpt_train"] = None

    yield "GPT 학습 완료!"


# ========== 6. TTS 추론 ==========
tts_pipeline = None


def load_tts_model(sovits_path, gpt_path):
    """TTS 모델 로드"""
    global tts_pipeline

    if not sovits_path or not gpt_path:
        return "SoVITS와 GPT 모델 경로를 모두 지정하세요"

    try:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        config_path = "GPT_SoVITS/configs/tts_infer.yaml"
        tts_config = TTS_Config(config_path)
        tts_pipeline = TTS(tts_config)

        # 모델 가중치 로드
        tts_pipeline.init_vits_weights(sovits_path)
        tts_pipeline.init_t2s_weights(gpt_path)

        return f"모델 로드 완료!\nSoVITS: {sovits_path}\nGPT: {gpt_path}"
    except Exception as e:
        traceback.print_exc()
        return f"모델 로드 실패: {str(e)}"


def run_tts(text, text_lang, ref_audio_path, prompt_text, prompt_lang,
            top_k=15, top_p=1.0, temperature=1.0, speed=1.0,
            ref_text_free=False, seed=-1):
    """TTS 추론 실행"""
    global tts_pipeline

    if tts_pipeline is None:
        return None, "먼저 모델을 로드하세요"

    if not text:
        return None, "합성할 텍스트를 입력하세요"

    if not ref_audio_path:
        return None, "참조 오디오를 지정하세요"

    try:
        req = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_audio_path,
            "prompt_text": "" if ref_text_free else prompt_text,
            "prompt_lang": prompt_lang.lower(),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "temperature": float(temperature),
            "speed_factor": float(speed),
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "fragment_interval": 0.3,
            "seed": int(seed),
            "streaming_mode": False,
            "return_fragment": False,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": 32,
            "super_sampling": False,
        }

        result = tts_pipeline.run(req)
        sr, audio_data = next(result)

        # numpy로 변환
        if isinstance(audio_data, np.ndarray):
            audio = audio_data
        else:
            audio = audio_data.cpu().numpy()

        return (sr, audio), "합성 완료!"

    except Exception as e:
        traceback.print_exc()
        return None, f"합성 실패: {str(e)}"


# ========== UI 구성 ==========
def refresh_model_list():
    sovits_names, gpt_names = scan_weights()
    return (
        gr.update(choices=sovits_names, value=sovits_names[0] if sovits_names else ""),
        gr.update(choices=gpt_names, value=gpt_names[0] if gpt_names else ""),
    )


def build_ui():
    gpu_info = "\n".join(GPU_INFOS)
    batch_size, batch_size_s1, sovits_epoch, save_every = get_version_defaults(VERSION)

    with gr.Blocks(
        title="GPT-SoVITS 음성 학습 & TTS",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# GPT-SoVITS 음성 학습 & TTS 서비스")
        gr.Markdown(f"GPU: {gpu_info} | Half precision: {is_half}")

        with gr.Tabs():
            # ===== 탭 1: 학습 =====
            with gr.TabItem("학습"):
                with gr.Accordion("모델 버전 설정", open=True):
                    version_select = gr.Dropdown(
                        choices=SUPPORTED_VERSIONS,
                        value=VERSION,
                        label="모델 버전",
                    )
                    exp_name = gr.Textbox(
                        label="실험 이름 (영문/숫자 권장)",
                        placeholder="my_voice_model",
                    )
                    gpu_number = gr.Textbox(
                        label="GPU 번호", value=DEFAULT_GPU,
                    )

                # Step 1: 오디오 슬라이싱
                with gr.Accordion("Step 1: 오디오 슬라이싱", open=False):
                    gr.Markdown("긴 오디오 파일을 짧은 구간으로 자동 분할합니다.")
                    with gr.Row():
                        slice_inp = gr.Textbox(
                            label="입력 오디오 경로 (파일 또는 폴더)",
                            placeholder="path/to/audio",
                        )
                        slice_out = gr.Textbox(
                            label="출력 폴더",
                            value="output/slicer_opt",
                        )
                    with gr.Row():
                        slice_threshold = gr.Slider(-60, 0, value=-34,
                                                     label="음량 임계값 (dB)")
                        slice_min_len = gr.Slider(1000, 10000, value=4000, step=100,
                                                   label="최소 길이 (ms)")
                        slice_min_interval = gr.Slider(100, 1000, value=300, step=10,
                                                        label="최소 간격 (ms)")
                    slice_btn = gr.Button("슬라이싱 시작", variant="primary")
                    slice_status = gr.Textbox(label="상태", interactive=False)
                    slice_stop_btn = gr.Button("중지")

                    slice_btn.click(
                        run_slice,
                        inputs=[slice_inp, slice_out, slice_threshold,
                                slice_min_len, slice_min_interval],
                        outputs=[slice_status],
                    )
                    slice_stop_btn.click(
                        lambda: stop_process("slice"),
                        outputs=[slice_status],
                    )

                # Step 2: 음성 인식 (ASR)
                with gr.Accordion("Step 2: 음성 인식 (ASR)", open=False):
                    gr.Markdown("슬라이싱된 오디오에서 텍스트를 자동 추출합니다.")
                    with gr.Row():
                        asr_inp = gr.Textbox(
                            label="입력 오디오 폴더",
                            placeholder="output/slicer_opt",
                        )
                        asr_out = gr.Textbox(
                            label="출력 폴더",
                            value="output/asr_opt",
                        )
                    with gr.Row():
                        asr_model = gr.Dropdown(
                            choices=["Faster Whisper (多语种)", "达摩 ASR (中文)"],
                            value="Faster Whisper (多语종)",
                            label="ASR 모델",
                        )
                        asr_size = gr.Dropdown(
                            choices=["medium", "large-v2", "large-v3", "large-v3-turbo"],
                            value="large-v3",
                            label="모델 크기",
                        )
                        asr_lang = gr.Dropdown(
                            choices=["ko", "ja", "en", "zh", "auto"],
                            value="ko",
                            label="언어",
                        )
                        asr_precision = gr.Dropdown(
                            choices=["float16", "float32", "int8"],
                            value="float16",
                            label="정밀도",
                        )
                    asr_btn = gr.Button("ASR 시작", variant="primary")
                    asr_status = gr.Textbox(label="상태", interactive=False)
                    asr_stop_btn = gr.Button("중지")

                    asr_btn.click(
                        run_asr,
                        inputs=[asr_inp, asr_out, asr_model, asr_size, asr_lang, asr_precision],
                        outputs=[asr_status],
                    )
                    asr_stop_btn.click(
                        lambda: stop_process("asr"),
                        outputs=[asr_status],
                    )

                # Step 3: 데이터셋 전처리
                with gr.Accordion("Step 3: 데이터셋 전처리 (1A+1B+1C)", open=False):
                    gr.Markdown("텍스트 분석, HuBERT 특성 추출, 시맨틱 토큰 추출을 순차 진행합니다.")
                    with gr.Row():
                        prep_text = gr.Textbox(
                            label="텍스트 리스트 파일 (.list)",
                            placeholder="output/asr_opt/xxx.list",
                        )
                        prep_wav_dir = gr.Textbox(
                            label="오디오 폴더 경로",
                            placeholder="output/slicer_opt",
                        )
                    prep_btn = gr.Button("전처리 시작", variant="primary")
                    prep_status = gr.Textbox(label="상태", interactive=False)
                    prep_stop_btn = gr.Button("중지")

                    prep_btn.click(
                        run_dataset_prep,
                        inputs=[version_select, prep_text, prep_wav_dir,
                                exp_name, gpu_number],
                        outputs=[prep_status],
                    )
                    prep_stop_btn.click(
                        lambda: stop_process("prep"),
                        outputs=[prep_status],
                    )

                # Step 4: SoVITS 학습
                with gr.Accordion("Step 4: SoVITS 학습", open=False):
                    with gr.Row():
                        sovits_batch = gr.Slider(1, 40, value=batch_size, step=1,
                                                  label="배치 사이즈")
                        sovits_epochs = gr.Slider(1, 25, value=sovits_epoch, step=1,
                                                   label="총 에폭")
                        sovits_save_every = gr.Slider(1, 25, value=save_every, step=1,
                                                       label="저장 간격 (에폭)")
                    sovits_train_btn = gr.Button("SoVITS 학습 시작", variant="primary")
                    sovits_status = gr.Textbox(label="상태", interactive=False)
                    sovits_stop_btn = gr.Button("중지")

                    sovits_train_btn.click(
                        run_sovits_train,
                        inputs=[version_select, exp_name, sovits_batch,
                                sovits_epochs, sovits_save_every],
                        outputs=[sovits_status],
                    )
                    sovits_stop_btn.click(
                        lambda: stop_process("sovits_train"),
                        outputs=[sovits_status],
                    )

                # Step 5: GPT 학습
                with gr.Accordion("Step 5: GPT 학습", open=False):
                    with gr.Row():
                        gpt_batch = gr.Slider(1, 40, value=batch_size_s1, step=1,
                                               label="배치 사이즈")
                        gpt_epochs = gr.Slider(1, 20, value=15, step=1,
                                                label="총 에폭")
                        gpt_save_every = gr.Slider(1, 10, value=5, step=1,
                                                    label="저장 간격 (에폭)")
                    gpt_train_btn = gr.Button("GPT 학습 시작", variant="primary")
                    gpt_status = gr.Textbox(label="상태", interactive=False)
                    gpt_stop_btn = gr.Button("중지")

                    gpt_train_btn.click(
                        run_gpt_train,
                        inputs=[version_select, exp_name, gpt_batch,
                                gpt_epochs, gpt_save_every],
                        outputs=[gpt_status],
                    )
                    gpt_stop_btn.click(
                        lambda: stop_process("gpt_train"),
                        outputs=[gpt_status],
                    )

            # ===== 탭 2: TTS 테스트 =====
            with gr.TabItem("TTS 테스트"):
                gr.Markdown("학습된 모델을 로드하고 음성을 합성합니다.")

                with gr.Accordion("모델 로드", open=True):
                    with gr.Row():
                        sovits_dropdown = gr.Dropdown(
                            choices=[], label="SoVITS 모델",
                        )
                        gpt_dropdown = gr.Dropdown(
                            choices=[], label="GPT 모델",
                        )
                        refresh_btn = gr.Button("모델 목록 새로고침")

                    load_btn = gr.Button("모델 로드", variant="primary")
                    load_status = gr.Textbox(label="상태", interactive=False)

                    refresh_btn.click(
                        refresh_model_list,
                        outputs=[sovits_dropdown, gpt_dropdown],
                    )
                    load_btn.click(
                        load_tts_model,
                        inputs=[sovits_dropdown, gpt_dropdown],
                        outputs=[load_status],
                    )

                with gr.Accordion("참조 오디오 설정", open=True):
                    ref_audio = gr.Audio(
                        label="참조 오디오 (3~10초 권장)",
                        type="filepath",
                    )
                    with gr.Row():
                        prompt_text = gr.Textbox(
                            label="참조 오디오 텍스트",
                            placeholder="참조 오디오의 대사 내용",
                        )
                        prompt_lang = gr.Dropdown(
                            choices=["ko", "zh", "en", "ja", "yue",
                                     "auto", "auto_yue"],
                            value="ko",
                            label="참조 오디오 언어",
                        )
                    ref_text_free = gr.Checkbox(
                        label="참조 텍스트 없이 사용 (품질 저하 가능)",
                        value=False,
                    )

                with gr.Accordion("텍스트 입력 & 합성", open=True):
                    tts_text = gr.Textbox(
                        label="합성할 텍스트",
                        placeholder="여기에 합성하고 싶은 텍스트를 입력하세요.",
                        lines=3,
                    )
                    with gr.Row():
                        text_lang = gr.Dropdown(
                            choices=["ko", "zh", "en", "ja", "yue",
                                     "auto", "auto_yue"],
                            value="ko",
                            label="텍스트 언어",
                        )
                        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                           label="속도")
                        seed = gr.Number(value=-1, label="시드 (-1=랜덤)")

                    with gr.Row():
                        top_k = gr.Slider(1, 100, value=15, step=1, label="Top K")
                        top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Top P")
                        temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.05,
                                                 label="Temperature")

                    tts_btn = gr.Button("음성 합성", variant="primary")
                    tts_output = gr.Audio(label="합성 결과", type="numpy")
                    tts_status = gr.Textbox(label="상태", interactive=False)

                    tts_btn.click(
                        run_tts,
                        inputs=[tts_text, text_lang, ref_audio, prompt_text,
                                prompt_lang, top_k, top_p, temperature,
                                speed, ref_text_free, seed],
                        outputs=[tts_output, tts_status],
                    )

        # 앱 로드 시 모델 목록 자동 새로고침
        app.load(refresh_model_list, outputs=[sovits_dropdown, gpt_dropdown])

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT-SoVITS Voice Service")
    parser.add_argument("--port", type=int, default=7860, help="서버 포트")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="서버 호스트")
    parser.add_argument("--share", action="store_true", help="Gradio 공유 링크 생성")
    parser.add_argument("--version", type=str, default="v2",
                        choices=SUPPORTED_VERSIONS, help="기본 모델 버전")
    args = parser.parse_args()

    VERSION = args.version
    os.environ["version"] = VERSION

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
