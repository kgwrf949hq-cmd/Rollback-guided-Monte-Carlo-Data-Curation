# Pick-and-place 第一版失败检测

## 范围

`pick_place_failure.py` 只检测失败并返回结构化结果，不捕获/选择 checkpoint，不调用 restore，不 replay 动作，也不直接终止环境。上层系统之后可以消费 `FailureDecision`，但检测器本身没有这些副作用。

## 输入和输出

输入：

- 当前 LIBERO observation；
- 当前 step、action、reward、done、info；
- 已完成的 pick-and-place 子事件列表。

输出：

```python
FailureDecision(
    status="warning",  # normal / warning / recoverable_failure / terminal_failure
    risk_score=0.88,
    reason="lift_no_progress",
    current_stage="object_lifted",
    consecutive_steps=2,
    hard_failure=False,
    evidence={"height_progress": 0.0, "required_progress": 0.015},
)
```

软失败先输出 `warning`，连续满足 `confirmation_steps` 后升级为 `recoverable_failure`。NaN、仿真异常、物体掉落、越界、阶段超时等硬规则立即触发；无效 observation 和未成功终止属于 `terminal_failure`。

## 第一版规则

| 当前阶段 | 规则 |
| --- | --- |
| approach | 距离窗口内无足够进展、阶段超时 |
| grasp | 夹爪已闭合但物体距离过远、阶段超时 |
| lift | 物体未升高、物体与末端分离、阶段超时 |
| transport/place | 到目标区域无进展、放置超时、物体掉落 |
| release | 夹爪打开后物体仍在目标区域外、阶段超时 |
| global | observation/action 非有限值、工作空间越界、仿真异常、done 但未成功 |

## 未来规则风险与模型风险融合

当前版本只有 `rule_risk`，尚未实现 `learned_risk`。未来若模型输出的 `learned_risk` 同样表示失败概率或失败风险，二者不能相减。可选方案包括：

```python
# 归一化加权和
final_risk = (
    rule_weight * rule_risk + model_weight * learned_risk
) / (rule_weight + model_weight)

# 保守融合
final_risk = max(rule_risk, learned_risk)

# 概率并集形式（需要校准，且隐含近似独立假设）
final_risk = 1.0 - (1.0 - rule_risk) * (1.0 - learned_risk)
```

第一版研究建议使用归一化加权和，同时让确定性硬规则拥有最高优先级；模型只参与软风险和提前预警。验证集上应分别报告规则、模型及融合结果的 precision、recall、检测延迟和误回退率。

## 验证

```bash
PYTHONPATH=. python validation/pick_place_failure_validation.py
```

该脚本覆盖正常接近、接近停滞、missed grasp、lift no progress、object dropped、placement timeout、released outside goal、invalid observation 和未成功终止。真实 LIBERO smoke 部分只检查 observation 兼容性与短轨迹误报，不涉及 checkpoint、restore 或 replay。
