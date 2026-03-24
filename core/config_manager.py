import json
from pathlib import Path
from typing import Any


class ConfigManager:
    # 该管理器负责统一读取平台配置，便于后续多平台扩展时保持加载入口一致。
    def __init__(self, project_root: Path | None = None) -> None:
        # 默认以当前文件上一级目录作为项目根目录。
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.platforms_dir = self.project_root / "prompt_library" / "platforms"

    # 根据平台名称加载对应 JSON 配置并返回字典结构。
    def load_platform_config(self, platform_name: str) -> dict[str, Any]:
        # 平台名规范化，避免中英文或大小写差异导致找不到文件。
        normalized_name = self._normalize_platform_name(platform_name)
        config_path = self.platforms_dir / f"{normalized_name}.json"

        # 未找到配置文件时抛出明确错误，提示调用方检查平台名或文件是否存在。
        if not config_path.exists():
            raise FileNotFoundError(
                f"未找到平台配置文件: {config_path}。请确认 platform_name 是否正确。"
            )

        try:
            with config_path.open("r", encoding="utf-8") as file:
                config_data = json.load(file)
        except json.JSONDecodeError as exc:
            # JSON 格式错误时抛出可读性更高的异常信息，便于快速定位配置问题。
            raise ValueError(f"平台配置文件 JSON 解析失败: {config_path}") from exc
        except OSError as exc:
            # 处理文件读取权限、I/O 异常等场景。
            raise OSError(f"读取平台配置文件失败: {config_path}") from exc

        if not isinstance(config_data, dict):
            raise TypeError(f"平台配置格式错误: {config_path} 顶层必须为 JSON 对象。")

        required_keys = {"platform_name", "pacing_rules", "style_guidelines", "banned_words"}
        missing_keys = required_keys - set(config_data.keys())
        if missing_keys:
            missing_keys_text = ", ".join(sorted(missing_keys))
            raise KeyError(f"平台配置缺少必要字段: {missing_keys_text}")

        return config_data

    # 将输入的平台名映射到标准文件名，便于调用层使用自然语言平台名。
    @staticmethod
    def _normalize_platform_name(platform_name: str) -> str:
        name = platform_name.strip().lower()
        alias_map = {
            "番茄": "tomato",
            "番茄小说": "tomato",
            "tomato": "tomato",
        }
        return alias_map.get(name, name.replace(" ", "_"))
