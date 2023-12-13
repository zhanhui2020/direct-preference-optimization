import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from copy import deepcopy

# Direct Preference Optimization的示例代码
# peter wu
# 2023-12-13

# DPO实际上并没有使用强化学习进行训练模型，而是采用了MLE的目标函数
# 通过policy model和reference model对于正负样本的差值进行训练
# reference model不进行梯度回传，只有policy model才进行梯度的计算

torch.manual_seed(0)

if __name__ == "__main__":
    # 超参数
    beta = 0.1
    # 加载模型
    policy_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=1000, num_hidden_layers=1, hidden_size=128))
    reference_model = deepcopy(policy_model)

    # data
    prompt_ids = [1, 2, 3, 4, 5, 6]
    good_response_ids = [7, 8, 9, 10]
    # 对loss稍加修改可以应对一个good和多个bad的情况
    bad_response_ids_list = [[1, 2, 3, 0], [4, 5, 6, 0]]

    # 转换成模型输入
    # tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    #         [ 1,  2,  3,  4,  5,  6,  1,  2,  3,  0],
    #         [ 1,  2,  3,  4,  5,  6,  4,  5,  6,  0]])
    input_ids = torch.LongTensor(
        [prompt_ids + good_response_ids, *[prompt_ids + bad_response_ids for bad_response_ids in bad_response_ids_list]]
    )

    # labels 提前做个shift
    # tensor([[-100, -100, -100, -100, -100,    7,    8,    9,   10],
    #         [-100, -100, -100, -100, -100,    1,    2,    3,    0],
    #         [-100, -100, -100, -100, -100,    4,    5,    6,    0]])
    labels = torch.LongTensor(
        [
            [-100] * len(prompt_ids) + good_response_ids,
            *[[-100] * len(prompt_ids) + bad_response_ids for bad_response_ids in bad_response_ids_list]
        ]
    )[:, 1:]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0
    # 计算 policy model的log prob
    logits = policy_model(input_ids)["logits"][:, :-1, :]
    # torch.gather从output_projection这层里面拿相应的答案那一列
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    all_logps = (per_token_logps * loss_mask).sum(-1)
    # 暂时写死第一个是good response的概率
    policy_good_logps, policy_bad_logps = all_logps[:1], all_logps[1:]

    # 计算 reference model的log prob
    with torch.no_grad():
        logits = reference_model(input_ids)["logits"][:, :-1, :]
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        all_logps = (per_token_logps * loss_mask).sum(-1)
        # 暂时写死第一个是good response的概率
        reference_good_logps, reference_bad_logps = all_logps[:1], all_logps[1:]

    # 计算loss，会自动进行广播
    # 这里可以看出，如果一个样本有多个偏序对的话，他是按照 (loss[good-bad1]+loss【good-bad2】)/2
    logits = (policy_good_logps - reference_good_logps) - (policy_bad_logps - reference_bad_logps)
    loss = -F.logsigmoid(beta * logits).mean()
    print(loss)
