"""이미지/영상 클립 처리 - Ken Burns 효과 및 AI 클립 후처리"""

import shlex
import subprocess
import sys


def apply_ken_burns(
    image_path: str,
    output_path: str,
    motion_type: str,
    duration: float,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
):
    """
    이미지에 Ken Burns 효과(줌/팬)를 적용하여 영상 클립 생성.

    4배 업스케일 후 zoompan 적용 → 줌 시 화질 저하 방지.
    """
    total_frames = int(duration * fps)
    zoom_speed = 0.25 / total_frames  # 전체 25% 줌

    # 업스케일 해상도 (4배) - 비율 유지하며 프레임을 채우도록 스케일링
    up_w = width * 4
    up_h = height * 4
    # 가로/세로 중 프레임을 채우는 쪽에 맞추고, 나머지는 비율 유지
    # -2는 ffmpeg에서 짝수 보장
    scale_expr = (
        f"scale='if(gt(iw/ih,{up_w}/{up_h}),{up_w},-2)'"
        f":'if(gt(iw/ih,{up_w}/{up_h}),-2,{up_h})'"
        f":flags=lanczos"
    )

    filter_map = {
        "zoom_in": (
            f"zoompan=z='min(1+{zoom_speed}*on,1.25)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        ),
        "zoom_out": (
            f"zoompan=z='max(1.25-{zoom_speed}*on,1.0)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        ),
        "pan_left": (
            f"zoompan=z='1.15':"
            f"x='(iw-iw/zoom)*(1-on/{total_frames})':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        ),
        "pan_right": (
            f"zoompan=z='1.15':"
            f"x='(iw-iw/zoom)*on/{total_frames}':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        ),
        "pan_up": (
            f"zoompan=z='1.15':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='(ih-ih/zoom)*(1-on/{total_frames})':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        ),
        "pan_down": (
            f"zoompan=z='1.15':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='(ih-ih/zoom)*on/{total_frames}':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        ),
    }

    zoompan_filter = filter_map.get(motion_type, filter_map["zoom_in"])
    vf = f"{scale_expr},{zoompan_filter}"

    cmd = (
        f'ffmpeg -y -loop 1 -i "{image_path}" '
        f'-vf "{vf}" '
        f"-t {duration} "
        f"-c:v libx264 -preset fast -crf 18 "
        f"-pix_fmt yuv420p -r {fps} -an "
        f'"{output_path}"'
    )

    if sys.platform == "win32":
        args = cmd
    else:
        args = shlex.split(cmd)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ken Burns 실패: {result.stderr[:500]}")

    return output_path


def process_ai_clip(
    clip_path: str,
    output_path: str,
    duration: float,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    zoom_start: float = 1.0,
    zoom_end: float = 1.05,
):
    """
    AI 생성 영상 클립에 trim + 서서히 줌인 효과 적용.

    - TTS 길이에 맞춰 클립을 잘라냄 (trim)
    - 매 프레임마다 scale을 점진적으로 키우고 중앙 crop (줌인)
    - zoom_start=1.0 → zoom_end=1.05 : 5% 서서히 줌인
    """
    zoom_range = zoom_end - zoom_start

    # 1. 먼저 목표 해상도로 업스케일 (AI 클립은 512x916 등 저해상도)
    # 2. 매 프레임 점진적 확대 (1.0x → 1.05x)
    # 3. 확대된 프레임의 정중앙을 목표 크기로 crop
    vf = (
        f"scale={width}:{height}:flags=lanczos,"
        f"scale=w='iw*({zoom_start}+{zoom_range}*t/{duration})':"
        f"h='ih*({zoom_start}+{zoom_range}*t/{duration})':eval=frame:flags=lanczos,"
        f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2"
    )

    cmd = (
        f'ffmpeg -y -i "{clip_path}" '
        f'-t {duration} '
        f'-vf "{vf}" '
        f"-c:v libx264 -preset fast -crf 18 "
        f"-pix_fmt yuv420p -r {fps} -an "
        f'"{output_path}"'
    )

    if sys.platform == "win32":
        args = cmd
    else:
        args = shlex.split(cmd)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"AI 클립 처리 실패: {result.stderr[:500]}")

    return output_path
