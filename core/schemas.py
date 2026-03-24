from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class NLPBaseTraits(BaseModel):
    environment: str = Field(
        ...,
        description="人物所处的外部环境设定，包含时代背景、社会结构、生活圈层与长期生存条件。",
    )
    behavior: str = Field(
        ...,
        description="人物的行为模式设定，描述其在压力、日常互动与关键抉择中的稳定行动倾向。",
    )
    capability: str = Field(
        ...,
        description="人物能力设定，覆盖知识、技能、资源与短板，体现其可完成与难完成的行动边界。",
    )
    values: str = Field(
        ...,
        description="人物价值观设定，说明其核心判断标准、底线原则与冲突中的优先级排序。",
    )
    identity: str = Field(
        ...,
        description="人物身份认同设定，包括自我定位、社会角色与他人视角下的标签及心理归属。",
    )
    vision: str = Field(
        ...,
        description="人物愿景设定，定义其长期目标、人生方向与驱动持续行动的终局想象。",
    )


class WorldCard(BaseModel):
    region_name: str = Field(
        ...,
        description="世界区域名称，用于标识故事主要发生地或关键地理板块。",
    )
    tech_level: str = Field(
        ...,
        description="该区域的科技或修真发展水平，用于约束能力上限与文明表现形态。",
    )
    power_structure: str = Field(
        ...,
        description="该区域的权力格局，描述官方、宗门、家族或帮派之间的统治关系。",
    )
    hidden_rules: str = Field(
        ...,
        description="该区域默认生效但不公开宣示的潜规则，影响角色行动风险与收益。",
    )


class FactionCard(BaseModel):
    faction_name: str = Field(
        ...,
        description="势力名称，用于在多线剧情中唯一标识该组织或阵营。",
    )
    core_philosophy: str = Field(
        ...,
        description="势力核心理念，说明其价值取向、行为原则与长期战略目标。",
    )
    leader: str = Field(
        ...,
        description="势力首脑名称或身份称谓，代表该势力当前最高决策者。",
    )
    attitude_to_protagonist: str = Field(
        ...,
        description="势力对主角的当前态度，通常为中立、敌对或拉拢等可演化立场。",
    )


class ItemCard(BaseModel):
    item_name: str = Field(
        ...,
        description="关键道具名称，用于在剧情推进中识别具体物品实体。",
    )
    origin: str = Field(
        ...,
        description="道具来历，描述其来源背景、流转历史或初次出现情境。",
    )
    current_owner: str = Field(
        ...,
        description="道具当前持有者，标识该物品在当前时间点的控制归属。",
    )
    hidden_power: str = Field(
        ...,
        description="道具尚未完全公开的隐藏功效，用于后续反转或能力升级铺垫。",
    )


class BookOutline(BaseModel):
    book_title: str = Field(
        ...,
        description="书名，用于标识全书主题定位与市场传播名称。",
    )
    logline: str = Field(
        ...,
        description="一句话简介，需浓缩主角目标、冲突核心与读者爽点承诺。",
    )
    core_power_system: str = Field(
        ...,
        description="全书核心力量体系或金手指设定，定义主角成长与世界规则边界。",
    )
    main_storyline: str = Field(
        ...,
        description="全书主线剧情脉络，描述从开局到终局的关键推进路径。",
    )
    ending_vision: str = Field(
        ...,
        description="预期大结局方向，说明最终冲突收束方式与主角终极状态。",
    )
    planned_volumes: List[str] = Field(
        ...,
        description="计划包含的卷名列表，按叙事推进顺序列出全书阶段结构。",
    )
    target_word_count: str = Field(
        ...,
        description="全书预计总字数规划，建议以万字或百万字口径表达，体现商业连载体量目标。",
    )
    pacing_design: str = Field(
        ...,
        description="全书节奏与高潮分布设计，说明起承转合、关键爆点与中后期拉升安排。",
    )


class ConceptProposal(BaseModel):
    core_hook: str = Field(
        ...,
        description="故事核心卖点，要求直接回答读者为什么会持续追更。",
    )
    golden_finger: str = Field(
        ...,
        description="主角关键金手指或核心能力设定，需体现成长空间与剧情驱动力。",
    )
    world_tone: str = Field(
        ...,
        description="世界基调，定义作品整体气质与叙事风格方向。",
    )


class WritingRule(BaseModel):
    rule_id: str = Field(
        ...,
        description="写作法则唯一标识，建议使用时间戳或 UUID。",
    )
    rule_name: str = Field(
        ...,
        description="法则名称，例如番茄黄金三章或极简短句排版。",
    )
    group: str = Field(
        default="通用法则 (General)",
        description="法则分组，用于在挂载舱中按主题归类展示。",
    )
    category: Literal["Elements", "Theories", "Taboos", "Formatting", "Lore", "Tropes"] = Field(
        ...,
        description="法则类别，限定为 Elements/Theories/Taboos/Formatting/Lore/Tropes。",
    )
    applicable_stage: Literal["全局", "总纲", "卷纲", "章纲", "正文"] = Field(
        ...,
        description="法则适用阶段。",
    )
    positive_instructions: List[str] = Field(
        ...,
        description="AI 必须执行的正向指导动作列表。",
    )
    negative_constraints: List[str] = Field(
        ...,
        description="AI 绝对禁止触犯的负向约束列表。",
    )


class VolumeOutline(BaseModel):
    volume_number: int = Field(
        ...,
        description="卷号，用于标识该分卷在全书结构中的顺序位置。",
    )
    volume_title: str = Field(
        ...,
        description="卷名，用于概括本卷主题与阶段性叙事目标。",
    )
    core_conflict: str = Field(
        ...,
        description="本卷核心冲突，定义该卷最关键的对立关系与主要矛盾。",
    )
    new_factions: List[str] = Field(
        ...,
        description="本卷新出场势力列表，用于明确新增博弈方及其剧情作用。",
    )
    estimated_word_count: str = Field(
        ...,
        description="本卷预计字数规划，体现阶段篇幅投入与连载节奏控制。",
    )
    key_subplots: List[str] = Field(
        ...,
        description="本卷核心支线剧情列表，需服务主线冲突并承担角色成长或世界扩展职责。",
    )


class ChapterOutlineList(BaseModel):
    volume_number: int = Field(
        ...,
        description="所属卷号，用于将单章连载大纲归档到对应分卷。",
    )
    chapters: List[Dict[str, Union[int, str]]] = Field(
        ...,
        description="单章脑洞列表，每个元素必须包含 chapter_number 与 core_event 两个字段。",
    )


class CharacterCardRecord(BaseModel):
    entity_id: str = Field(
        ...,
        description="角色唯一标识符，用于在多章节、多版本设定中稳定引用同一角色实体。",
    )
    version_chapter: int = Field(
        ...,
        description="该设定版本开始生效的章节号，表示此人格与行为基线从该章节起进入时间轴。",
    )
    traits: NLPBaseTraits = Field(
        ...,
        description="角色在当前版本下的六层人物设定快照，作为章节生成与一致性校验的基础输入。",
    )
    update_reason: Optional[str] = Field(
        default="初始设定",
        description="角色性格或设定发生变化的原因说明，用于追踪人设演化逻辑与记忆更新时间线。",
    )


class ChapterBeatTemplate(BaseModel):
    chapter_number: int = Field(
        ...,
        description="章节编号，用于定位当前单章节拍器模板对应的叙事节点。",
    )
    beat_1_setup: str = Field(
        ...,
        description="前 300 字铺垫与起因约束，负责交代情境、目标与基础矛盾触发条件。",
    )
    beat_2_conflict: str = Field(
        ...,
        description="中间 800 字冲突与打压约束，推动对抗升级并持续施压主角行动空间。",
    )
    beat_3_climax: str = Field(
        ...,
        description="中间 700 字高潮与反击约束，要求矛盾集中爆发并给出关键逆转或强对抗结果。",
    )
    beat_4_cliffhanger: str = Field(
        ...,
        description="末尾 200 字悬念约束，在阶段性收束后保留强钩子以驱动下一章节阅读。",
    )
