(function () {
  "use strict";

  const STORAGE_KEY = "motionbuilder-studio-language";
  const RIG_STORAGE_KEY = "mka-lang";

  const EXACT = Object.freeze({
    "正在检查": "CHECKING",
    "正在读取发布能力…": "Checking release capabilities...",
    "状态：检查中": "STATUS:CHECK",
    "正在检查工作台服务与发布能力。": "Checking the workbench service and release capabilities.",
    "实验性 · 公开版暂不提供": "EXPERIMENTAL · UNAVAILABLE IN PUBLIC RELEASE",
    "实验性，后续会更新与HybrIK以及GVHMR的兼容包": "Experimental. Compatibility packages for HybrIK and GVHMR are planned for a future update.",
    "配置未就绪": "CONFIGURATION NOT READY",
    "发布配置未就绪，暂不启用新的图片或视频推理。已有资产功能不受影响。": "Release configuration is not ready. New image/video inference is disabled; existing asset features are unaffected.",
    "服务未就绪": "SERVICE NOT READY",
    "工作台服务尚未就绪，暂不能新建推理任务。": "The workbench service is not ready. New inference tasks are disabled for now.",
    "研究 / 实验性": "RESEARCH / EXPERIMENTAL",
    "本地研究模式：此入口按已配置的研究后端运行，资源与许可需分别满足。": "Local research mode: this entry uses the configured research backend. Resource and permission requirements apply separately.",
    "在线 · 实验性功能": "ONLINE · EXPERIMENTAL",
    "工作台在线；公开版暂不启用图片与视频新推理，其它功能按各自资源与依赖条件使用。": "Workbench online. New image/video inference is not enabled in this public release; other features retain their own resource and dependency requirements.",
    "正在加载已配置的 HybrIK 研究后端": "Loading configured HybrIK research backend",
    "HybrIK 单图推理": "HybrIK single-image inference",
    "正在将原始结果转换为 SOMA30": "Converting original output to SOMA30",
    "HybrIK 独立研究进程推理中": "HybrIK independent research process running",
    "图片 Pose 任务已取消": "Image Pose task cancelled",
    "状态：错误": "STATUS:ERROR",
    "状态：在线": "STATUS:ONLINE",
    "在线 · 可选后端未配置": "ONLINE · OPTIONAL NOT READY",
    "工作台服务无法连接或尚未就绪。": "Workbench service is unreachable or not ready.",
    "工作台在线；部分可选后端尚未配置，已有资产编辑仍可使用。": "Workbench online; some optional backends are not configured. Existing asset editing remains available.",
    "工作台服务在线；后端文件状态不代表模型运行或许可验证。": "Workbench service online; backend file status is not model execution or license verification.",
    "导演仓库": "Director Warehouse",
    "导演场景创建与编辑": "Director Scene Creation & Editing",
    "新建导演工程": "New Director Project",
    "刷新导演仓库": "Refresh Director Warehouse",
    "导演资产仓库管理": "Director Asset Management",
    "导演与镜头构建": "Director & Shot Builder",
    "导演工程名称": "Director project name",
    "未命名导演工程": "Untitled Director Project",
    "撤销": "Undo",
    "重做": "Redo",
    "版本快照": "Version Snapshot",
    "保存工程": "Save Project",
    "场景组件": "Scene Components",
    "导入骨骼网格体动画": "Import Skeletal Mesh Animation",
    "动画资产": "Animation Asset",
    "网格体动画": "Mesh Animation",
    "暂无骨骼网格体": "No skeletal meshes",
    "该网格体暂无动画": "No animations for this skeletal mesh",
    "启用根运动（显示轨迹）": "Enable Root Motion (show trajectory)",
    "添加人物动画": "Add Character Animation",
    "导入场景资产": "Import Scene Asset",
    "资产类型": "Asset Type",
    "静态网格体": "Static Mesh",
    "地形": "Terrain",
    "添加到场景": "Add to Scene",
    "增加机位": "Add Camera",
    "增加相机轨道": "Add Camera Path",
    "编辑视角": "Editor View",
    "导演视角": "Director View",
    "框选选中物体": "Frame Selected",
    "框选全部人物": "Frame All Characters",
    "画幅": "Aspect",
    "总帧数": "Total Frames",
    "输出": "Output",
    "代理": "Proxy",
    "添加 Camera Cut": "Add Camera Cut",
    "添加黑幕片段": "Add Blackout Segment",
    "添加时间轴标记": "Add Timeline Marker",
    "添加局部变速": "Add Local Time Warp",
    "添加相机关键帧": "Add Camera Key",
    "删除当前关键帧": "Delete Current Key",
    "刷新镜头缩略图": "Refresh Shot Thumbnails",
    "属性": "Properties",
    "在场景大纲或时间轴中选择对象": "Select an object in the outliner or timeline",
    "构图辅助": "Composition Guides",
    "安全框": "Safe Frame",
    "中心线": "Center Lines",
    "三分线": "Rule of Thirds",
    "导出": "Export",
    "导出当前截图": "Export Current Screenshot",
    "导出导演视频": "Export Director Video",
    "导出分镜联系表": "Export Contact Sheet",
    "导出参考包": "Export Reference Package",
    "深度图": "Depth Pass",
    "法线图": "Normal Pass",
    "人物/场景分割图": "Character/Scene Segmentation",
    "取消导出": "Cancel Export",
    "继续导出": "Resume Export",
    "航迹 Pose 与预览场景编辑": "Trajectory Pose & Preview Scene Editing",
    "生成高级设置": "Advanced Generation Settings",
    "文本引导强度": "Text Guidance Strength",
    "约束引导强度": "Constraint Guidance Strength",
    "多提示词过渡帧数": "Multi-Prompt Transition Frames",
    "初始朝向 / 度": "Initial Heading / Degrees",
    "原生后处理": "Native Post-Processing",
    "逐帧稠密根轨迹": "Dense Per-Frame Root Path",
    "根部修正边距 / 米": "Root Correction Margin / m",
    "约束创作": "Constraint Authoring",
    "当前航迹点姿势编辑": "Current Waypoint Pose Editing",
    "骨骼节点": "Bone Node",
    "恢复库中姿势": "Restore Library Pose",
    "修改仅保存在本次生成设置中，不会覆盖姿势仓库资产。": "Changes are kept only in this generation setup and never overwrite the pose-library asset.",
    "选择手或脚的 Two-Bone IK 控制点，在预测帧区间内创建空间约束；开始帧与结束帧相同即为单帧约束。": "Select a hand or foot Two-Bone IK control and create a world-space constraint over the predicted frame range; matching start and end values create a single-frame constraint.",
    "IK 控制点": "IK Control",
    "生成 IK 控制": "Create IK Control",
    "复位控制点": "Reset Control",
    "Two-Bone IK 约束：生成控制点后进行平移 / 旋转": "Two-Bone IK constraint: create a control, then translate / rotate it.",
    "Two-Bone IK 约束：选择控制点后进行平移 / 旋转": "Two-Bone IK constraint: select a control, then translate / rotate it.",
    "所选 Pose 不包含可编辑的全身姿势约束": "The selected pose has no editable full-body pose constraint",
    "FK 骨骼": "FK Bones",
    "约束类型": "Constraint Type",
    "编辑方式": "Editing Method",
    "全身关键帧": "Full-Body Keyframe",
    "左手末端": "Left-Hand Effector",
    "右手末端": "Right-Hand Effector",
    "左脚末端": "Left-Foot Effector",
    "右脚末端": "Right-Foot Effector",
    "组合末端": "Mixed Effectors",
    "全身 IK": "Full Body IK",
    "左手": "Left Hand",
    "右手": "Right Hand",
    "左脚": "Left Foot",
    "右脚": "Right Foot",
    "开始帧": "Start Frame",
    "结束帧": "End Frame",
    "新建 / 更新约束": "Create / Update Constraint",
    "新约束": "New Constraint",
    "新建约束": "New Constraint",
    "保存约束": "Save Constraint",
    "更新约束": "Update Constraint",
    "生成失败": "Generation Failed",
    "Kimodo 生成失败": "Kimodo Generation Failed",
    "删除约束": "Delete Constraint",
    "平移": "Translate",
    "旋转": "Rotate",
    "吸附当前姿势": "Snap to Current Pose",
    "复位编辑": "Reset Editing",
    "FK 骨骼 / IK 控制点": "FK Bone / IK Control",
    "导出约束": "Export Constraints",
    "导入约束": "Import Constraints",
    "预览场景": "Preview Scene",
    "资产类型": "Asset Type",
    "资产": "Asset",
    "静态网格体": "Static Mesh",
    "地形": "Terrain",
    "添加到预览": "Add to Preview",
    "整体缩放": "Uniform Scale",
    "复位": "Reset",
    "删除": "Delete",
    "约束编辑：选择 FK 骨骼或 IK 控制点后进行平移 / 旋转": "Constraint editing: select an FK bone or IK control, then translate / rotate.",
    "脚部锁定": "Foot Lock",
    "源动画": "Source Animation",
    "基准帧": "Reference Frame",
    "锁定脚": "Locked Feet",
    "全身 IK 解算器": "Full-Body IK Solver",
    "根节点策略": "Root Strategy",
    "双脚": "Both Feet",
    "脚部优先": "Feet First",
    "根节点预拉": "Root Pre-Pull",
    "固定根节点": "Pin Root",
    "自由根节点": "Free Root",
    " 同时锁定脚部旋转": " Lock Foot Rotation Too",
    "选定帧的脚部世界状态作为基准；处理后虚拟根固定在该帧双脚中点，朝向逐帧跟随 Hips。": "The selected frame defines the foot world-space targets; the rebuilt virtual root stays at that frame's foot midpoint and follows the Hips heading per frame.",
    "生成脚部锁定预览": "Generate Foot-Lock Preview",
    "帧": "Frames",
    "暂无可用动画": "No Animations Available",
    "选择基准帧和锁定方式后生成预览；源动画不会被覆盖。": "Choose a reference frame and lock mode, then generate a preview; the source animation is not overwritten.",
    "所选角色暂无可用动画。": "The selected character has no available animations.",
    "最大锁定误差": "Maximum Lock Error",
    "平均误差": "Average Error",
    "不可达帧": "Unreachable Frames",
    "虚拟根漂移": "Virtual Root Drift",
    "正在逐帧执行全身 IK 与专用虚拟根重算…": "Solving full-body IK per frame and rebuilding the dedicated virtual root…",
    "\u6574\u4f53\u901f\u5ea6": "Overall Speed",
    "\u6574\u4f53\u901f\u5ea6\u5feb\u6377\u8bbe\u7f6e": "Overall Speed Presets",
    "模型仓库": "Model Library",
    "预览控制": "Preview Controls",
    "一键复位预览": "Reset Preview",
    "模型预览：请选择骨骼并进行平移或旋转。": "Model preview: select a bone, then translate or rotate it.",
    "IK 模式：拖动控制器验证姿势。": "IK mode: drag a controller to validate the pose.",
    "模型预览已复位。": "Model preview reset.",
    "姿势仓库": "Pose Library",
    "动作仓库": "Motion Library",
    "角色绑定与蒙皮": "Rigging & Skinning",
    "自动IK": "Auto IK",
    "IK 求解器": "IK Solver",
    "IK 解算器": "IK Solver",
    "全身 IK": "Full Body IK",
    "操作": "Transform",
    "移动": "Translate",
    "局部": "Local",
    "世界": "World",
    "根骨骼行为": "Root Behavior",
    "预拉伸": "Pre-pull",
    "固定到输入姿势": "Pin to Input Pose",
    "自由": "Free",
    "迭代次数": "Iterations",
    "确认IK并保存角色": "Confirm IK & Save Character",
    "预览模式": "Preview Mode",
    "IK控制": "IK Controls",
    "未选择IK控制器": "No IK control selected",
    "双臂 · 双腿": "Arms · Legs",
    "嘴部与眼球已排除": "Jaw and eye joints excluded",
    "重置IK预览": "Reset IK Preview",
    "IK：已绑定": "IK: Bound",
    "IK：未绑定": "IK: Not Bound",
    "视频动作推理": "Video-to-Motion",
    "文本动作生成": "Text-to-Motion",
    "动作重定向": "Motion Retargeting",
    "最终动画构建": "Final Animation Builder",
    "拖动调整总体时间轴高度": "Drag to Resize Master Timeline Height",
    "资产导入与预处理": "Asset Import & Preprocessing",
    "绑定新角色": "Rig New Character",
    "导入静态资产": "Import Static Asset",
    "模型资产仓库管理": "Model Asset Library Management",
    "全部资产": "All Assets",
    "角色资产": "Character Assets",
    "静态资产": "Static Assets",
    "上传视频": "Upload Video",
    "开始动作推理": "Start Motion Inference",
    "视频动作仓库": "Video Motion Library",
    "姿势导入与生成": "Pose Import & Generation",
    "图片姿势推理": "Image Pose Inference",
    "图片预览": "Image Preview",
    "导入已有姿势": "Import Existing Pose",
    "从动作提取姿势": "Extract Pose from Motion",
    "文本生成动作": "Text-Generated Motion",
    "视频推理动作": "Video-Inference Motion",
    "暂无可预览动作": "No Previewable Motions",
    "Pose 名称": "Pose Name",
    "当前帧": "Current Frame",
    "导出当前帧": "Export Current Frame",
    "姿势资产仓库管理": "Pose Asset Library Management",
    "动作名称": "Motion Name",
    "添加提示词片段": "Add Prompt Segment",
    "打开航迹控制": "Enable Trajectory Control",
    "添加航迹点": "Add Waypoint",
    "推理步数": "Inference Steps",
    "随机种子": "Random Seed",
    "开始生成": "Start Generation",
    "文本动作仓库": "Text Motion Library",
    "平移模式": "Translate Mode",
    "旋转模式": "Rotate Mode",
    "只迁移父子局部旋转；固定 Hips 与所有目标骨长，绝不拉伸骨骼。": "Transfer only local parent-child rotations; keep Hips and all target bone lengths fixed—never stretch bones.",
    "动作来源": "Motion Source",
    "源动作": "Source Motion",
    "目标骨骼网格体": "Target Skeletal Mesh",
    "生成重定向预览": "Generate Retarget Preview",
    "添加至动作库": "Add to Motion Library",
    "为动作生成统一的 ground / locomotion root：World → VirtualRoot → Hip / Pelvis。预览确认后再存为新的动作库条目。": "Create a unified ground / locomotion root for the motion: World → VirtualRoot → Hip / Pelvis. Preview and confirm before saving it as a new Motion Library entry.",
    "新动作名称": "New Motion Name",
    "平滑帧数": "Smoothing Frames",
    "静止容差 / 帧": "Idle Tolerance / Frame",
    "当前流程按项目标准 Y-up 计算。VirtualRoot 位于脚底最低基准面，Hip / Pelvis 的局部起伏与摆动会保留在其下方。": "This workflow uses the project's standard Y-up coordinate system. VirtualRoot sits on the lowest foot baseline, while the local rise, fall, and sway of Hip / Pelvis are preserved beneath it.",
    "生成 VirtualRoot 预览": "Generate VirtualRoot Preview",
    "确认并存入动作库": "Confirm and Save to Motion Library",
    "动作文件处理": "Motion File Processing",
    "动画合成": "Animation Compositing",
    "动画调整": "Animation Adjustment",
    "VIRTUALROOT 构建": "VIRTUALROOT Builder",
    "动作资产仓库管理": "Motion Asset Library Management",
    "全部动作": "All Motions",
    "普通动作": "Standard Motions",
    "加法动画叠加": "Additive Animation",
    "全身加法动画叠加": "Full-Body Additive Animation",
    "加法叠加动画": "Additive Composite",
    "基础动画（决定输出长度）": "Base Animation (Controls Output Length)",
    "加法动画层": "Additive Animation Layer",
    "基础动画速率": "Base Animation Rate",
    "加法层速率": "Additive Layer Rate",
    "叠加强度": "Additive Weight",
    "层结束行为": "Layer End Behavior",
    "循环": "Loop",
    "保持最后一帧": "Hold Last Frame",
    "参考姿势": "Reference Pose",
    "加法动画首帧": "Additive First Frame",
    "加法动画指定帧": "Specified Additive Frame",
    "骨骼绑定姿势": "Skeleton Bind Pose",
    "参考帧": "Reference Frame",
    "输出长度跟随调速后的基础动画；加法层会按所选速率循环或保持。": "Output length follows the rate-adjusted base animation; the additive layer loops or holds at its selected rate.",
    "生成加法叠加预览": "Generate Additive Preview",
    "存为新动画": "Save as New Animation",

    "接触": "Contact",
    "摆动": "Swing",
    "IK 步态动画": "IK Gait Animation",
    "IK 步态、贴地与滑步": "IK Gait, Grounding & Foot Lock",
    "IK 步态、贴地与滑步控制": "IK Gait, Grounding & Foot-Lock Control",
    "带根运动的动画": "Root-Motion Animation",
    "步幅倍率": "Stride Scale",
    "根运动路程倍率": "Root Travel Scale",
    "脚锁定强度": "Foot-Lock Strength",
    "接触高度阈值 / 米": "Contact Height Threshold / m",
    "接触速度阈值 / 米每秒": "Contact Speed Threshold / m/s",
    "种植混合帧": "Plant Blend Frames",
    "鞋底偏移 / 米": "Sole Offset / m",
    "骨盆补偿上限 / 米": "Pelvis Compensation Limit / m",
    "骨盆平滑帧": "Pelvis Smoothing Frames",
    "脚部旋转贴合地面": "Align Foot Rotation to Ground",
    "处理前会把根运动垂直位置归一到 Y=0 地面基准；原动画不会被覆盖。": "Before processing, root-motion height is normalized to the Y=0 ground plane; the original animation is never overwritten.",
    "生成 IK 步态预览": "Generate IK Gait Preview",
    "左脚：-- · 右脚：--": "Left Foot: -- · Right Foot: --",
    "当前帧用于确定整段动画的局部修正量。": "The current frame defines the local correction applied across the animation.",
    "重置 IK 预览": "Reset IK Preview",
    "Two-Bone IK · 双臂、双腿": "Two-Bone IK · Arms and Legs",
    "全身 IK · 头部、双手、双脚": "Full Body IK · Head, Hands, and Feet",
    "当前动画没有关联骨骼网格体": "The current animation has no associated skeletal mesh.",
    "自动 IK Profile 生成失败": "Automatic IK Profile generation failed.",
    "未选择 IK 控制器": "No IK control selected",
    "IK 控制模式：可平移或旋转 Goal；Pole 仅支持平移。": "IK Control Mode: translate or rotate a Goal; Pole controls support translation only.",
    "FK 骨骼模式：调整会应用到整段动画。": "FK Bone Mode: adjustments are applied across the animation.",
    "IK 预览已重置": "IK preview reset.",
    "正在载入动画与 IK Profile…": "Loading animation and IK Profile…",
    "所选骨骼网格体暂无带根运动的动画。": "The selected skeletal mesh has no root-motion animations.",
    "调整参数后生成预览；结果只会另存为新动画。": "Adjust the parameters and generate a preview; the result can only be saved as a new animation.",
    "正在进行根运动归一、脚步接触检测与 IK 求解…": "Normalizing root motion, detecting foot contacts, and solving IK…",
    "正在另存为新动画…": "Saving as a new animation…",
    "请先载入骨骼网格体": "Load a skeletal mesh first.",
    "IK 控制模式：拖动或旋转控制器，调整将按当前作用范围提交。": "IK Control Mode: translate or rotate controls; adjustments use the current scope.",
    "骨骼模式：请选择骨骼并进行平移或旋转调整。": "Bone Mode: select a bone, then translate or rotate it.",
    "IK 预览已重置；已保存的骨骼修复不受影响。": "IK preview reset; saved bone repairs are unchanged.",
    "骨骼修复模式：可切换骨骼或 IK，并进行平移、旋转调整。": "Bone Repair Mode: switch between Bone and IK controls for translation and rotation.",
    "动作导入模式": "Motion Import Mode",
    "片段处理模式": "Clip Processing Mode",
    "组件添加模式": "Component Add Mode",
    "时间轴构建": "Timeline Builder",
    "选择角色": "Select Character",
    "选择动画": "Select Animation",
    "预览所选动作": "Preview Selected Motion",
    "添加到时间轴": "Add to Timeline",
    "退出动作导入模式": "Exit Motion Import Mode",
    "删除开始帧（包含）": "Delete Start Frame (Inclusive)",
    "删除结束帧（不包含）": "Delete End Frame (Exclusive)",
    "删除帧": "Delete Frames",
    "截断动画": "Trim Animation",
    "退出片段处理模式": "Exit Clip Processing Mode",
    "静态网格体": "Static Mesh",
    "添加组件到选中骨骼": "Add Component to Selected Bone",
    "平移": "Translate",
    "旋转": "Rotate",
    "缩放": "Scale",
    "整体缩放": "Uniform Scale",
    "确认组件": "Confirm Components",
    "退出但不应用": "Exit Without Applying",
    "骨骼修复模式": "Bone Repair Mode",
    "骨骼": "Bone",
    "平移骨骼": "Translate Bone",
    "旋转骨骼": "Rotate Bone",
    "局部轴": "Local Axes",
    "世界轴": "World Axes",
    "作用范围": "Scope",
    "当前关键帧": "Current Keyframe",
    "帧区间": "Frame Range",
    "整段动画": "Entire Animation",
    "开始帧": "Start Frame",
    "结束帧": "End Frame",
    "前过渡帧": "Pre-Transition Frames",
    "后过渡帧": "Post-Transition Frames",
    "局部位移（米）": "Local Translation (m)",
    "局部旋转（度）": "Local Rotation (deg)",
    "添加修复": "Add Repair",
    "清除修复": "Clear Repairs",
    "退出骨骼修复模式": "Exit Bone Repair Mode",
    "点击骨骼节点选择 · 拖拽 Gizmo 调整 · 滚轮缩放": "Click a bone node to select · Drag the Gizmo to adjust · Scroll to zoom",

    "动作预览": "Motion Preview",
    "关闭": "Close",
    "视频预览": "Video Preview",
    "根运动": "Root Motion",
    "世界运动": "World-Space Motion",
    "锁定根骨骼": "Lock Root Bone",
    "焦距 mm（可选）": "Focal Length mm (Optional)",
    "静态相机视频": "Static-Camera Video",
    "开始推理": "Start Inference",
    "取消上传": "Cancel Upload",
    "取消导入": "Cancel Import",
    "进行推理": "Run Inference",
    "重命名": "Rename",
    "姿势编辑": "Pose Editing",
    "选中骨骼": "Selected Bone",
    "新 Pose 名称": "New Pose Name",
    "保存为新 Pose": "Save as New Pose",
    "取消": "Cancel",
    "骨骼网格体": "Skeletal Mesh",
    "上半身动画": "Upper-Body Animation",
    "下半身动画": "Lower-Body Animation",
    "合成总时长 / 秒": "Total Composite Duration / sec",
    "新动画名称": "New Animation Name",
    "生成合成预览": "Generate Composite Preview",
    "存入动作库": "Save to Motion Library",
    "生成VIRTUALROOT": "Generate VIRTUALROOT",
    "确认并替换原动作": "Confirm and Replace Original Motion",
    "动作文件": "Motion File",
    "坐标系": "Coordinate Space",
    "局部坐标": "Local Space",
    "世界坐标": "World Space",
    "复位所选骨骼": "Reset Selected Bone",
    "从当前帧截断并丢弃前段": "Trim at Current Frame and Discard Earlier Section",
    "从当前帧截断并丢弃后段": "Trim at Current Frame and Discard Later Section",
    "撤回": "Undo",
    "存为新动作": "Save as New Motion",
    "覆盖原动作": "Overwrite Original",

    "静态": "Static",
    "角色": "Character",
    "姿势": "Pose",
    "普通": "Standard",
    ".静态": ".Static",
    ".角色": ".Character",
    ".姿势": ".Pose",
    ".普通": ".Standard",
    "未指定角色": "No Character Assigned",
    "未命名动作": "Untitled Motion",
    "全部角色": "All Characters",
    "暂无资产": "No Assets Yet",
    "暂无动作资产": "No Animation Assets",
    "暂无视频动作": "No Video Motions",
    "暂无文本动作": "No Text Motions",
    "暂无姿势。可导入已有姿势，或从动作提取姿势。": "No poses yet. Import an existing pose or extract a pose from motion.",
    "预览": "Preview",
    "编辑": "Edit",
    "删除": "Delete",
    "下载": "Download",
    "导出 FBX": "Export FBX",
    "导出 constraints": "Export Constraints",
    "正在终止": "Terminating",
    "终止生成": "Stop Generation",
    "终止中": "Terminating",
    "处理中": "Processing",
    "正在上传…": "Uploading…",
    "动作": "Motion",
    "视频动作": "Video Motion",
    "合成动画": "Composite Animation",
    "静态资产预览": "Static Asset Preview",
    "角色预览": "Character Preview",
    "拖动片段排序": "Drag to Reorder Segments",
    "提示词输入": "Prompt",
    "片段时长": "Segment Duration",
    "删除片段": "Delete Segment",
    "时间": "Time",
    "位移": "Distance",
    "启用姿势控制": "Pose Control",
    "姿势选择": "Pose Selection",
    "不使用": "None",
    "删除航迹点": "Delete Waypoint",
    "整体调整模式": "Overall Adjustment Mode",
    "曲线编辑模式": "Curve Edit Mode",
    "平移曲线": "Translate Curves",
    "旋转曲线": "Rotate Curves",
    "取消所有选择": "Clear All Selections",
    "选中曲线首尾组合": "Connect Selected Curves End-to-End",
    "所有曲线自动对齐": "Auto-align All Curves",
    "应用编辑": "Apply Edits",
    "选择整体": "Select Entire Curve",
    "取消选择": "Clear Selection",
    "控制点数量": "Control Point Count",
    "平滑为控制点曲线": "Convert to Smooth Control-Point Curve",
    "压平 X": "Flatten X",
    "压平 Y": "Flatten Y",
    "压平 Z": "Flatten Z",
    "应用曲线编辑": "Apply Curve Edits",
    "朝向修正模式": "Orientation Correction Mode",
    "添加至动作库": "Add to Animation Library",
    "模式": "Mode",
    "单帧": "Single Frame",
    "应用修正": "Apply Correction",
    "整体 Root 曲线": "Entire Root Curve",
    "无根运动": "No Root Motion",
    "曲线位置": "Curve Position",
    "曲线旋转": "Curve Rotation",
    "整体": "Entire Curve",
    "片段": "Segment",
    "未选择": "Not Selected",
    "无静态网格体": "No Static Mesh",
    "附着物": "Attachment",
    "位置": "Position",
    "选择": "Select",
    "合并片段": "Merge Segments",
    "添加过渡帧": "Add Transition Frames",
    "入": "In",
    "出": "Out",
    "速度": "Speed",
    "反向": "Reverse",
    "镜像": "Mirror",
    "应用变换": "Apply Transform",
    "删除朝向修正": "Delete Orientation Correction",
    "应用朝向修正": "Apply Orientation Correction",
    "删除过渡": "Delete Transition",
    "应用过渡": "Apply Transition",
    "前过渡": "Transition In",
    "后过渡": "Transition Out",
    "修复": "Repair",
    "片段过渡": "Segment Transition",
    "朝向修正": "Orientation Correction",
    "朝向修正前过渡": "Orientation Correction Transition In",
    "朝向修正后过渡": "Orientation Correction Transition Out",
    "暂无骨骼修复": "No Bone Repairs Yet",
    "已有视频动作正在生成": "A Video Motion Is Already Being Generated",
    "已有文本动作正在生成": "A Text Motion Is Already Being Generated",
    "缺少权重或 SMPL 模型": "Missing Weights or an SMPL Model",
    "尚未安装代码": "Code Not Installed",
    "最终动画": "Final Animation",
    "VirtualRoot 动作": "VirtualRoot Motion",
    "动画片段": "Animation Segment",
    "当前选定控制点：未选择": "Selected control point: None",

    "Pose 尚未载入完成，不能保存。": "Pose has not finished loading and cannot be saved.",
    "正在保存新的 Pose…": "Saving New Pose…",
    "正在导入静态资产…": "Importing Static Asset…",
    "请先选择一张图片。": "Select an image first.",
    "图片姿势推理失败": "Image pose inference failed",
    "图片姿势推理等待超时": "Image pose inference timed out",
    "正在导入已有姿势…": "Importing Existing Pose…",
    "正在载入动作列表…": "Loading Motion List…",
    "正在载入动作…": "Loading Motion…",
    "请先选择并载入一个动作文件。": "Select and load a motion file first.",
    "请先填写 Pose 名称。": "Enter a Pose name first.",
    "正在写入姿势库…": "Saving to the Pose Library…",
    "没有可撤回的操作": "Nothing to undo.",
    "裁切后至少需要保留 1 帧": "At least 1 frame must remain after trimming.",
    "当前帧没有可裁切的片段": "No segment can be trimmed at the current frame.",
    "正在载入动画…": "Loading Animation…",
    "正在覆盖原动作…": "Overwriting Original Motion…",
    "正在保存新动作…": "Saving New Motion…",
    "已覆盖原动作": "Original motion overwritten.",
    "已存为新动作": "Saved as a new motion.",
    "航迹控制至少需要 2 个航迹点": "Trajectory control requires at least 2 waypoints.",
    "航迹点时间必须按顺序递增": "Waypoint times must increase in sequence.",
    "航迹点已打开 Pose，但尚未选择姿势仓库中的 Pose": "Pose is enabled for a waypoint, but no Pose has been selected from the Pose Library.",
    "请至少填写一个提示词片段": "Enter at least one prompt segment.",
    "合成中": "Compositing…",
    "预览已生成": "Preview Generated",
    "正在生成加法叠加预览…": "Generating Additive Preview…",
    "正在生成重定向预览": "Generating Retarget Preview…",
    "正在生成 VirtualRoot 预览…": "Generating VirtualRoot Preview…",
    "正在构建VIRTUALROOT": "Building VIRTUALROOT…",
    "正在存入动作库…": "Saving to Animation Library…",
    "正在添加至动作库…": "Adding to Animation Library…",
    "正在替换原动作…": "Replacing Original Motion…",
    "3D 预览尚未初始化完成": "3D preview has not finished initializing.",
    "动作导入模式：选择角色和动画后可预览或添加到时间轴。": "Animation Import Mode: select a character and animation, then preview or add it to the timeline.",
    "已退出动作导入模式。": "Exited Animation Import Mode.",
    "片段处理模式：可删除指定帧区间或在当前帧截断动画。": "Segment Processing Mode: delete a frame range or split the animation at the current frame.",
    "已退出片段处理模式。": "Exited Segment Processing Mode.",
    "骨骼修复模式：请选择骨骼并进行局部轴或世界轴调整。": "Bone Repair Mode: select a bone, then adjust it in local or world axes.",
    "已退出骨骼修复模式。": "Exited Bone Repair Mode.",
    "整体调整模式：请选择彩色曲线段或空心三角，并整体平移/旋转。": "Overall Adjustment Mode: select colored curve segments or hollow triangles, then translate/rotate them as a group.",
    "已退出整体调整模式，未应用更改。": "Exited Overall Adjustment Mode without applying changes.",
    "朝向修正模式：只能拖动时间轴预览，并在左侧选择修正区间。": "Orientation Correction Mode: scrub the timeline preview and select the correction range on the left.",
    "已退出朝向修正模式，未应用更改。": "Exited Orientation Correction Mode without applying changes.",
    "曲线编辑模式：请先在左侧选择整体或一个可编辑动画片段。": "Curve Edit Mode: select the entire curve or an editable animation segment on the left.",
    "已取消曲线编辑。": "Curve editing canceled.",
    "已取消所有曲线选择。": "Cleared all curve selections.",
    "已取消曲线选择": "Curve selection cleared.",
    "已取消 Root Curve 选择。": "Root Curve selection cleared.",
    "请至少选择两段曲线再进行首尾组合。": "Select at least two curves before connecting them end-to-end.",
    "已按时间轴顺序首尾组合选中曲线；未选中的片段保持原位置。": "Connected selected curves end-to-end in timeline order; unselected segments stayed in place.",
    "所有曲线已按当前旋转自动对齐：片段 1 不动，后续片段起点贴合上一段终点。": "Auto-aligned all curves using their current rotation: segment 1 stayed fixed, and each later segment starts at the previous segment’s end.",
    "正在逐帧应用分段 Root 曲线和根旋转…": "Applying segmented Root Curves and root rotation frame by frame…",
    "曲线分段整体调整已应用：Root 位置与旋转已按当前分段曲线逐帧烘焙。": "Overall segmented-curve adjustments applied: Root position and rotation were baked frame by frame from the current segmented curves.",
    "已选择整体 Root 曲线，并载入上次整体控制点。": "Selected the entire Root Curve and loaded its last saved control points.",
    "已选择整体 Root 曲线。点击“平滑为控制点曲线”后可编辑。": "Selected the entire Root Curve. Click “Convert to Smooth Control-Point Curve” to edit it.",
    "请先选择整体或片段并生成控制点。": "Select the entire curve or a segment and generate control points first.",
    "请先选择整体或一个可编辑动画片段。": "Select the entire curve or an editable animation segment first.",
    "请先点击“平滑为控制点曲线”生成并确认要编辑的控制点。": "Click “Convert to Smooth Control-Point Curve” first to generate and confirm the control points to edit.",
    "正在按片段原始路程比例烘焙选中 Root Curve…": "Baking the selected Root Curve using the segment's original path-length ratio…",
    "正在按弧长重映射并烘焙整体 Root Curve…": "Remapping by arc length and baking the entire Root Curve…",
    "选中片段不可编辑": "The selected segment cannot be edited.",
    "选中片段 Root Curve 已按原始路程比例应用；其它片段未被编辑。": "The selected segment's Root Curve was applied using its original path-length ratio; other segments were not edited.",
    "Root Curve 已应用，并已替换当前时间轴中的旧曲线。": "Root Curve applied and the old curve on the current timeline was replaced.",
    "正在应用朝向修正…": "Applying Orientation Correction…",
    "朝向修正已应用：仅修正 VirtualRoot yaw，位置未改变。": "Orientation correction applied: only VirtualRoot yaw was corrected; position was unchanged.",
    "请选择一个静态网格体": "Select a static mesh.",
    "当前骨骼网格体不可用": "The current bone mesh is unavailable.",
    "请先在预览视口中选择一个骨骼节点。": "Select a bone node in the preview viewport first.",
    "点击预览视口中的骨骼节点，然后选择静态网格体并附着。": "Click a bone node in the preview viewport, then select and attach a static mesh.",
    "组件已应用到当前预览。": "Components applied to the current preview.",
    "已退出组件添加模式，未应用本次改动。": "Exited Component Add Mode without applying these changes.",
    "当前该模块仅支持VIRTUALROOT动画": "This module currently supports VIRTUALROOT animations only.",
    "动画预览与修复仅允许添加带 VirtualRoot 的动作。": "Animation preview and repair only accept motions with VirtualRoot.",
    "当前动作与时间轴角色不一致，请先清空时间轴后再更换角色。": "The current motion's character does not match the timeline character. Clear the timeline before changing characters.",
    "只能合并连续的动画片段": "Only consecutive animation segments can be merged.",
    "只能为相邻的两个动画片段添加过渡": "Transitions can only be added between two adjacent animation segments.",
    "片段过渡已应用至时间轴": "Segment transition applied to the timeline.",
    "朝向修正已应用至时间轴": "Orientation correction applied to the timeline.",
    "正在删除片段过渡…": "Deleting segment transition…",
    "已删除片段过渡": "Segment transition deleted.",
    "正在删除朝向修正…": "Deleting orientation correction…",
    "已删除朝向修正": "Orientation correction deleted.",
    "动画已更新": "Animation Updated",
    "正在按 VIRTUALROOT 世界坐标重新合成…": "Recompositing in VIRTUALROOT world coordinates…",
    "时间轴为空": "The timeline is empty.",
    "正在删除帧并重新组合…": "Deleting frames and recompositing…",
    "尚未调整任何骨骼": "No bones have been adjusted.",
    "请先向时间轴添加动画片段。": "Add an animation segment to the timeline first.",
    "请先将动作添加到时间轴并载入骨骼网格体。": "Add a motion to the timeline and load the bone mesh first.",
    "请先将带 VirtualRoot 的动作添加到时间轴。": "Add a motion with VirtualRoot to the timeline first.",
    "请先在时间轴内添加至少一段动画，并载入骨骼网格体。": "Add at least one animation segment to the timeline and load the bone mesh first.",
    "请先退出当前曲线/朝向模式。": "Exit the current curve/orientation mode first.",
    "动作名称不能为空": "Motion name cannot be empty.",
    "片段名称不能为空": "Segment name cannot be empty.",
    "结束帧必须大于开始帧（结束帧不包含在删除范围内）": "End frame must be greater than start frame (the end frame is excluded from the deletion range).",

    "新的角色名称": "New Character Name",
    "新的 Pose 名称": "New Pose Name",
    "新的动作名称": "New Motion Name",
    "新的视频动作名称": "New Video Motion Name",
    "新的文本动作名称": "New Text Motion Name",
    "姿势名称": "Pose Name",
    "输入动作名称": "Enter Motion Name",
    "重命名动画片段": "Rename Animation Segment",
    "删除该静态资产？": "Delete this static asset?",
    "删除角色以及其动作库和时间轴数据？": "Delete this character, its animation library, and timeline data?",
    "删除该 Pose？": "Delete this Pose?",
    "删除该动作？": "Delete this motion?",
    "删除该视频动作？此操作无法撤回。": "Delete this video motion? This cannot be undone.",
    "删除该文本动作？此操作无法撤回。": "Delete this text motion? This cannot be undone.",
    "终止该视频动作生成任务？": "Stop this video motion generation task?",
    "终止该文本动作生成任务？": "Stop this text motion generation task?",
    "确定覆盖原动作？此操作会替换该动作文件。": "Overwrite the original motion? This will replace its motion file.",
    "确定用当前 VIRTUALROOT 结果替换原动作？此操作会覆盖原动作文件。": "Replace the original motion with the current VIRTUALROOT result? This will overwrite its motion file.",
    "清除当前动画的全部骨骼修复？": "Clear all bone repairs for the current animation?",

    "上传未绑定网格体": "Upload Unrigged Mesh",
    "重新上传": "Re-upload",
    "确认上传": "Confirm Upload",
    "自动对齐": "Auto Align",
    "重设变换": "Reset Transform",
    "确认对齐": "Confirm Alignment",
    "旋转（度）": "Rotation (deg)",
    "开始骨架绑定": "Start Skeleton Rigging",
    "确认骨架": "Confirm Skeleton",
    "确认变换": "Confirm Transform",
    "确认蒙皮并保存角色": "Confirm Skinning and Save Character",
    "确认蒙皮": "Confirm Skin",
    "重设姿态": "Reset Pose",
    "模型透明度": "Mesh Opacity",
    "未选择骨骼": "No Bone Selected",
    "蒙皮权重修正": "Skin Weight Correction",
    "增加权重": "Add Weight",
    "擦除权重": "Erase Weight",
    "取消工具": "Cancel Tool",
    "画笔大小": "Brush Size",
    "画笔力度": "Brush Strength",
    "下载 binding.npz": "Download binding.npz",
    "自动骨骼生成中": "Generating Skeleton Automatically",
    "自动蒙皮推理中": "Automatic Skinning Inference",
    "将模型与红色骨骼大致对齐后确认，可提高后续流程精度": "Roughly align the model with the red skeleton, then confirm to improve downstream accuracy.",
    "已应用自动对齐，请检查后确认": "Auto alignment applied. Review and confirm.",
    "微调骨骼位置后确认进入蒙皮": "Fine-tune bone positions, then confirm to begin skinning.",
    "变换已重设": "Transform reset.",
    "绘制模式：增加权重": "Paint Mode: Add Weight",
    "擦除模式：降低权重": "Erase Mode: Reduce Weight",
    "工具取消：点选骨骼来验证姿态": "Tool Canceled: Select a bone to verify the pose.",
    "姿态已重设": "Pose reset.",
    "前": "F",
    "侧": "S",
    "顶": "T",
    "透": "P"
  });

  const RULES = [
    [/^\u6574\u4f53\u901f\u5ea6 ([\d.]+)\u00d7(?: \u00b7 (\d+) \u5e27 \u2192 (\d+) \u5e27 \u00b7 ([\d.]+) \u79d2)?$/u, (m) => `Overall Speed ${m[1]}\u00d7${m[2] ? ` \u00b7 ${m[2]} frames \u2192 ${m[3]} frames \u00b7 ${m[4]} sec` : ""}`],
    [/^IK \u6b65\u6001\u9884\u89c8\u5df2\u751f\u6210\uff1a(\d+) \u5e27 \u00b7 ([\d.]+)\u00d7\uff1b\u6839\u8fd0\u52a8\u9ad8\u5ea6\u5df2\u56fa\u5b9a\u5728 Y=0\u3002$/u, (m) => `IK gait preview generated: ${m[1]} frames \u00b7 ${m[2]}\u00d7; root-motion height is fixed at Y=0.`],
    [/^IK 已求解，待保存调整：(\d+) 个骨骼$/u, (m) => `IK solved; unsaved adjustments: ${m[1]} bones.`],
    [/^IK 已求解，待提交 (\d+) 根骨骼的调整。$/u, (m) => `IK solved; pending adjustments for ${m[1]} bones.`],
    [/^滑步距离 ([\d.]+) m → ([\d.]+) m$/u, (m) => `Foot-slide distance ${m[1]} m → ${m[2]} m`],
    [/^改善 ([\d.]+)%$/u, (m) => `Improvement ${m[1]}%`],
    [/^左脚接触 ([\d.]+)% · 右脚接触 ([\d.]+)%$/u, (m) => `Left contact ${m[1]}% · Right contact ${m[2]}%`],
    [/^左脚：(接触|摆动) (\d+)% · 右脚：(接触|摆动) (\d+)%$/u, (m) => `Left: ${m[1] === "接触" ? "Contact" : "Swing"} ${m[2]}% · Right: ${m[3] === "接触" ? "Contact" : "Swing"} ${m[4]}%`],
    [/^IK 步态预览已生成：(\d+) 帧；根运动高度已固定在 Y=0。$/u, (m) => `IK gait preview generated: ${m[1]} frames; root-motion height is fixed at Y=0.`],
    [/^删除失败：(.+)$/u, (m) => `Delete failed: ${m[1]}`],
    [/^重命名失败：(.+)$/u, (m) => `Rename failed: ${m[1]}`],
    [/^终止失败：(.+)$/u, (m) => `Stop failed: ${m[1]}`],
    [/^导入失败：(.+)$/u, (m) => `Import failed: ${m[1]}`],
    [/^3D 初始化失败：(.+)$/u, (m) => `3D initialization failed: ${m[1]}`],
    [/^HybrIK 状态检查失败：(.+)$/u, (m) => `HybrIK status check failed: ${m[1]}`],
    [/^HybrIK 运行时未就绪：(.+)$/u, (m) => `HybrIK runtime is not ready: ${toEnglish(m[1])}`],
    [/^Pose 预览载入失败：(.+)$/u, (m) => `Failed to load Pose preview: ${m[1]}`],
    [/^默认 SOMA 骨架加载失败：(.+)$/u, (m) => `Failed to load default SOMA skeleton: ${m[1]}`],
    [/^(.+) 预览加载失败：(.+)$/u, (m) => `Failed to load ${m[1]} preview: ${m[2]}`],
    [/^生成进度 (\d+)%$/u, (m) => `Generation progress ${m[1]}%`],
    [/^进度 (\d+)%$/u, (m) => `Progress ${m[1]}%`],
    [/^图片 Pose 预览 · (.+)$/u, (m) => `Image Pose Preview · ${m[1]}`],
    [/^姿势仓库预览 · (.+)$/u, (m) => `Pose Library Preview · ${m[1]}`],
    [/^姿势编辑 · (.+)$/u, (m) => `Pose Editor · ${m[1]}`],
    [/^Kimodo 原生动作预览 · (.+)$/u, (m) => `Kimodo Native Motion Preview · ${m[1]}`],
    [/^静态资产预览 · (.+)$/u, (m) => `Static Asset Preview · ${m[1]}`],
    [/^角色预览 · (.+)$/u, (m) => `Character Preview · ${m[1]}`],
    [/^动作预览 · (.+)$/u, (m) => `Motion Preview · ${m[1]}`],
    [/^视频预览 · (.+)$/u, (m) => `Video Preview · ${m[1]}`],
    [/^已保存：(.+)$/u, (m) => `Saved: ${m[1]}`],
    [/^已加载动作预览：(.+)$/u, (m) => `Motion preview loaded: ${m[1]}`],
    [/^动作预览载入失败：(.+)$/u, (m) => `Failed to load motion preview: ${m[1]}`],
    [/^名称保存失败：(.+)$/u, (m) => `Failed to save the name: ${m[1]}`],
    [/^动作生成中：(.+?) · 总时长 ([\d.]+)s$/u, (m) => `Generating motion: ${m[1]} · Total duration ${m[2]}s`],
    [/^已存入动作库：(.+)$/u, (m) => `Saved to Animation Library: ${m[1]}`],
    [/^预览已生成：(.+)$/u, (m) => `Preview generated: ${m[1]}`],
    [/^预览已生成 · (\d+) 帧 · ([\d.]+) fps$/u, (m) => `Preview generated · ${m[1]} frames · ${m[2]} fps`],
    [/^已存为新动画：(.+)$/u, (m) => `Saved as new animation: ${m[1]}`],
    [/^已添加至动作库：(.+)$/u, (m) => `Added to Animation Library: ${m[1]}`],
    [/^VirtualRoot 预览已生成：(.+)。确认后将作为特殊动作存入动作库。$/u, (m) => `VirtualRoot preview generated: ${m[1]}. Confirm to save it as a special motion in the Animation Library.`],
    [/^已生成：(.+)$/u, (m) => `Generated: ${m[1]}`],
    [/^已替换原动作：(.+)$/u, (m) => `Original motion replaced: ${m[1]}`],
    [/^已撤回，当前保留 (\d+) 帧，待保存调整：(\d+) 个骨骼$/u, (m) => `Undone. ${m[1]} frames remain; unsaved adjustments: ${m[2]} bones.`],
    [/^已裁切，当前保留 (\d+) 帧$/u, (m) => `Trimmed; ${m[1]} frames remain.`],
    [/^已选择骨骼：(.+)$/u, (m) => `Selected bone: ${m[1]}`],
    [/^待保存调整：(\d+) 个骨骼$/u, (m) => `Unsaved adjustments: ${m[1]} bones`],
    [/^已复位：(.+)$/u, (m) => `Reset: ${m[1]}`],
    [/^片段 (\d+)$/u, (m) => `Segment ${m[1]}`],
    [/^(.+?) · (\d+) 帧 · (VirtualRoot|普通动作)$/u, (m) => `${m[1]} · ${m[2]} frames · ${m[3] === "普通动作" ? "Standard Motion" : m[3]}`],
    [/^(\d+)\. (.+?) · (\d+) 帧( · 无根运动)?$/u, (m) => `${m[1]}. ${m[2]} · ${m[3]} frames${m[4] ? " · No Root Motion" : ""}`],
    [/^已选择 (\d+) 段曲线；组合枢轴：(.+) 起点$/u, (m) => `Selected ${m[1]} curve segments; group pivot: start of ${m[2]}`],
    [/^朝向修正预览区间：(\d+) – (\d+)$/u, (m) => `Orientation correction preview range: ${m[1]}–${m[2]}`],
    [/^已选择片段：(.+)，并载入上次保存的控制点。$/u, (m) => `Selected segment: ${m[1]}, and loaded its last saved control points.`],
    [/^已选择片段：(.+)。点击“平滑为控制点曲线”后只编辑该片段。$/u, (m) => `Selected segment: ${m[1]}. Click “Convert to Smooth Control-Point Curve” to edit only this segment.`],
    [/^当前选定控制点：(\d+)$/u, (m) => `Selected control point: ${m[1]}`],
    [/^Root Curve 已压平 ([XYZ]) 轴。$/u, (m) => `Root Curve flattened on the ${m[1]} axis.`],
    [/^待提交骨骼调整：(\d+)$/u, (m) => `Pending bone adjustments: ${m[1]}`],
    [/^已选择骨骼：(.+)。选择静态网格体后点击“添加组件到选中骨骼”。$/u, (m) => `Selected bone: ${m[1]}. Select a static mesh, then click “Add Component to Selected Bone”.`],
    [/^正在调整：(.+) → (.+)$/u, (m) => `Adjusting: ${m[1]} → ${m[2]}`],
    [/^已附着：(.+) → (.+)。可用 Gizmo 调整位置\/旋转\/整体缩放。$/u, (m) => `Attached: ${m[1]} → ${m[2]}. Use the Gizmo to adjust position, rotation, and uniform scale.`],
    [/^位置 (.+) · 缩放 ([\d.-]+)$/u, (m) => `Position ${m[1]} · Scale ${m[2]}`],
    [/^加载 (\d+)\/(\d+)$/u, (m) => `Loading ${m[1]}/${m[2]}`],
    [/^动画已就绪：(\d+)帧$/u, (m) => `Animation ready: ${m[1]} frames`],
    [/^已合并 (\d+) 段$/u, (m) => `Merged ${m[1]} segments`],
    [/^(\d+)\. 已合并 (\d+) 段$/u, (m) => `${m[1]}. Merged ${m[2]} segments`],
    [/^片段已重命名：(.+)$/u, (m) => `Segment renamed: ${m[1]}`],
    [/^片段过渡：前 (\d+) 帧 \+ 后 (\d+) 帧$/u, (m) => `Segment transition: ${m[1]} frames in + ${m[2]} frames out`],
    [/^(前|后)过渡[： ](\d+) 帧$/u, (m) => `${m[1] === "前" ? "Transition In" : "Transition Out"}: ${m[2]} frames`],
    [/^朝向修正过渡：前 (\d+) 帧 \+ 后 (\d+) 帧$/u, (m) => `Orientation correction transition: ${m[1]} frames in + ${m[2]} frames out`],
    [/^朝向修正(前|后)过渡 · (\d+) 帧$/u, (m) => `Orientation correction transition ${m[1] === "前" ? "in" : "out"} · ${m[2]} frames`],
    [/^片段过渡 · 前 (\d+) 帧 \+ 后 (\d+) 帧$/u, (m) => `Segment transition · ${m[1]} frames in + ${m[2]} frames out`],
    [/^朝向修正 · (\d+)–(\d+)$/u, (m) => `Orientation correction · ${m[1]}–${m[2]}`],
    [/^已删除 (\d+) 段失效的过渡帧$/u, (m) => `Removed ${m[1]} invalid transition ranges`],
    [/^已删除 \[(\d+), (\d+)\)，剩余 (\d+) 帧(?: · 已删除 (\d+) 段失效的过渡帧)?$/u, (m) => `Deleted [${m[1]}, ${m[2]}); ${m[3]} frames remain${m[4] ? ` · removed ${m[4]} invalid transition ranges` : ""}`],
    [/^正在第 (\d+) 帧截断动画…$/u, (m) => `Splitting animation at frame ${m[1]}…`],
    [/^已在第 (\d+) 帧截断动画(?: · 已删除 (\d+) 段失效的过渡帧)?$/u, (m) => `Animation split at frame ${m[1]}${m[2] ? ` · removed ${m[2]} invalid transition ranges` : ""}`],
    [/^正在提交 (\d+) 根骨骼的调整…$/u, (m) => `Submitting adjustments for ${m[1]} bones…`],
    [/^(\d+) · (.+?) · (\d+) 帧$/u, (m) => `${m[1]} · ${m[2]} · ${m[3]} frames`],
    [/^(.+?) · (\d+) 帧( · VirtualRoot)?$/u, (m) => `${m[1]} · ${m[2]} frames${m[3] || ""}`],
    [/^(\d+) 帧$/u, (m) => `${m[1]} frames`]
  ];

  const VALUE_IDS = Object.freeze({
    retargetName: ["重定向动作", "Retargeted Motion"],
    virtualRootName: ["VirtualRoot 动作", "VirtualRoot Motion"],
    upperLowerName: ["合成动画", "Composite Animation"],
    additiveName: ["加法叠加动画", "Additive Composite"],
    gaitIKName: ["IK 步态动画", "IK Gait Animation"],
    libraryVirtualRootName: ["VirtualRoot 动作", "VirtualRoot Motion"]
  });

  const MANAGED_OPTION_SELECTS = new Set([
    "retargetSourceKind", "poseFromMotionKind", "editScope", "orientationScope", "animationAdjustSpace",
    "animationAdjustAnimation", "animationAdjustSolver", "finalIKSolver", "upperLowerUpper", "upperLowerLower",
    "additiveBase", "additiveLayer", "additiveReferenceType", "additiveEndBehavior",
    "libraryVirtualRootAnimation", "gaitIKCharacter", "gaitIKAnimation",
    "kimodoSceneKind"
  ]);

  const USER_DATA_SELECTORS = [
    ".asset-name", ".motion-card-name", ".kimodo-card-name", ".kimodo-card-error", ".pose-job-diagnostic",
    "[data-asset-preview-name]", "textarea", ".prompt-input"
  ].join(",");

  const originalText = new WeakMap();
  const originalAttributes = new WeakMap();
  const dirtyValueInputs = new WeakSet();
  const observers = new Map();
  let applyingValues = false;
  let language = readInitialLanguage();

  function readInitialLanguage() {
    try { return localStorage.getItem(STORAGE_KEY) === "en" ? "en" : "zh"; }
    catch (_) { return "zh"; }
  }

  function toEnglish(value) {
    const source = String(value == null ? "" : value);
    if (Object.prototype.hasOwnProperty.call(EXACT, source)) return EXACT[source];
    for (const [pattern, render] of RULES) {
      const match = source.match(pattern);
      if (match) return render(match);
    }
    return source;
  }

  function translatePreservingWhitespace(value) {
    const source = String(value == null ? "" : value);
    const match = source.match(/^(\s*)(.*?)(\s*)$/su);
    if (!match || !match[2]) return source;
    return `${match[1]}${toEnglish(match[2])}${match[3]}`;
  }

  function isUserDataNode(node) {
    const element = node.nodeType === 1 ? node : node.parentElement;
    if (!element) return false;
    if (element.closest(".studio-language-switch")) return true;
    if (element.closest(USER_DATA_SELECTORS)) return true;
    if (element.closest('[data-animation-character-filter]:not([data-animation-character-filter="all"])')) return true;
    if (element.closest(".pose-task-row strong")) return true;
    if (element.tagName === "OPTION") {
      const select = element.closest("select");
      if (!element.value) return false;
      return !select || !MANAGED_OPTION_SELECTS.has(select.id) && select.getAttribute("name") !== "root_mode";
    }
    return false;
  }

  function processTextNode(node, lang, fresh) {
    if (!node || node.nodeType !== 3 || isUserDataNode(node)) return;
    if (lang === "zh") {
      if (originalText.has(node)) node.nodeValue = originalText.get(node);
      return;
    }
    if (fresh || !originalText.has(node)) originalText.set(node, node.nodeValue);
    node.nodeValue = translatePreservingWhitespace(originalText.get(node));
  }

  function processElementAttributes(element, lang, fresh) {
    if (!element || element.nodeType !== 1 || element.closest(".studio-language-switch")) return;
    const attributes = ["title", "placeholder", "aria-label", "alt"];
    let originals = originalAttributes.get(element);
    if (!originals) { originals = Object.create(null); originalAttributes.set(element, originals); }
    for (const name of attributes) {
      if (!element.hasAttribute(name)) continue;
      if (lang === "zh") {
        if (Object.prototype.hasOwnProperty.call(originals, name)) element.setAttribute(name, originals[name]);
      } else {
        if (fresh || !Object.prototype.hasOwnProperty.call(originals, name)) originals[name] = element.getAttribute(name);
        element.setAttribute(name, toEnglish(originals[name]));
      }
    }
  }

  function walk(root, lang, fresh) {
    if (!root) return;
    if (root.nodeType === 3) { processTextNode(root, lang, fresh); return; }
    if (root.nodeType !== 1 && root.nodeType !== 9 && root.nodeType !== 11) return;
    if (root.nodeType === 1) processElementAttributes(root, lang, fresh);
    const doc = root.ownerDocument || root;
    const walker = doc.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (node.nodeType === 3) processTextNode(node, lang, fresh);
      else processElementAttributes(node, lang, fresh);
    }
  }

  function runWithoutObserver(doc, callback) {
    const observer = observers.get(doc);
    if (observer) observer.disconnect();
    try { callback(); }
    finally { if (observer) observeDocument(doc, observer); }
  }

  function observeDocument(doc, existing) {
    const observer = existing || new MutationObserver((records) => {
      runWithoutObserver(doc, () => {
        for (const record of records) {
          if (language === "en") {
            if (record.type === "characterData") processTextNode(record.target, "en", true);
            else if (record.type === "attributes") processElementAttributes(record.target, "en", true);
            else for (const node of record.addedNodes) walk(node, "en", true);
          } else if (record.type === "characterData" && !isUserDataNode(record.target)) {
            originalText.set(record.target, record.target.nodeValue);
          } else if (record.type === "attributes") {
            let originals = originalAttributes.get(record.target);
            if (!originals) { originals = Object.create(null); originalAttributes.set(record.target, originals); }
            originals[record.attributeName] = record.target.getAttribute(record.attributeName);
          }
        }
      });
    });
    observer.observe(doc.documentElement || doc, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: ["title", "placeholder", "aria-label", "alt"]
    });
    observers.set(doc, observer);
    return observer;
  }

  function applyValues(lang) {
    applyingValues = true;
    for (const [id, pair] of Object.entries(VALUE_IDS)) {
      const input = document.getElementById(id);
      if (!input || dirtyValueInputs.has(input)) continue;
      if (lang === "en" && input.value === pair[0]) input.value = pair[1];
      else if (lang === "zh" && input.value === pair[1]) input.value = pair[0];
    }
    applyingValues = false;
  }

  function updateLanguageControls() {
    document.documentElement.lang = language === "en" ? "en" : "zh-CN";
    document.documentElement.dataset.language = language;
    document.getElementById("studioLanguageZh")?.classList.toggle("active", language === "zh");
    document.getElementById("studioLanguageEn")?.classList.toggle("active", language === "en");
    document.getElementById("studioLanguageZh")?.setAttribute("aria-pressed", String(language === "zh"));
    document.getElementById("studioLanguageEn")?.setAttribute("aria-pressed", String(language === "en"));
  }

  function syncRigFrame(lang) {
    let frame;
    try { frame = document.getElementById("rigFrame"); } catch (_) { return; }
    if (!frame) return;
    try {
      const childWindow = frame.contentWindow;
      const childDocument = frame.contentDocument;
      if (!childWindow || !childDocument?.documentElement) return;
      if (lang === "zh") {
        runWithoutObserver(childDocument, () => walk(childDocument, "zh", false));
        childWindow.MKA_I18N?.setLang("zh");
      } else {
        childWindow.MKA_I18N?.setLang("en");
        if (!observers.has(childDocument)) observeDocument(childDocument);
        runWithoutObserver(childDocument, () => walk(childDocument, "en", false));
      }
      childDocument.documentElement.lang = lang === "en" ? "en" : "zh-CN";
    } catch (_) { /* The editor may still be loading; its load handler retries. */ }
  }

  function setLanguage(next, options) {
    language = next === "en" ? "en" : "zh";
    try {
      localStorage.setItem(STORAGE_KEY, language);
      localStorage.setItem(RIG_STORAGE_KEY, language);
    } catch (_) { /* Storage can be unavailable in hardened browser profiles. */ }
    const observer = observers.get(document);
    runWithoutObserver(document, () => {
      walk(document, language, false);
      applyValues(language);
      updateLanguageControls();
    });
    syncRigFrame(language);
    if (!options?.silent) window.dispatchEvent(new CustomEvent("motionbuilder:languagechange", { detail: { language } }));
    return language;
  }

  function installDialogTranslations() {
    const nativeAlert = window.alert.bind(window);
    const nativeConfirm = window.confirm.bind(window);
    const nativePrompt = window.prompt.bind(window);
    window.alert = (message) => nativeAlert(language === "en" ? toEnglish(message) : message);
    window.confirm = (message) => nativeConfirm(language === "en" ? toEnglish(message) : message);
    window.prompt = (message, defaultValue) => nativePrompt(language === "en" ? toEnglish(message) : message, defaultValue);
  }

  function init() {
    installDialogTranslations();
    for (const id of Object.keys(VALUE_IDS)) {
      document.getElementById(id)?.addEventListener("input", (event) => {
        if (!applyingValues) dirtyValueInputs.add(event.currentTarget);
      });
    }
    document.getElementById("studioLanguageZh")?.addEventListener("click", () => setLanguage("zh"));
    document.getElementById("studioLanguageEn")?.addEventListener("click", () => setLanguage("en"));
    const rigFrame = document.getElementById("rigFrame");
    rigFrame?.addEventListener("load", () => syncRigFrame(language));
    observeDocument(document);
    setLanguage(language, { silent: true });
  }

  window.MotionBuilderI18n = Object.freeze({
    getLanguage: () => language,
    setLanguage,
    translate: (value) => language === "en" ? toEnglish(value) : String(value == null ? "" : value),
    translateToEnglish: toEnglish,
    apply: (root) => runWithoutObserver(root?.ownerDocument || document, () => walk(root || document, language, false))
  });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
