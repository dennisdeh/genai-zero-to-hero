## Fine-tuning
### Full fine-tuning
#### Full fine-tuning of pre-trained MiniGPT
The model we in a previous chapter pre-trained can readily be fine-tuned for various tasks.

### Parameter-efficient fine-tuning
Full fine-tuning updates every weight of the model. For a model with billions of parameters this means storing
the weights, their gradients and the optimiser state (two extra moments for Adam) for all of them, which quickly
requires several times the memory of the model itself, and every fine-tuned variant is a full copy of the model.
Parameter-efficient fine-tuning (PEFT) instead freezes the pre-trained weights and trains only a small number of
additional parameters. The most popular PEFT method is LoRA (Low-Rank Adaptation, Hu et al. 2021).

#### LoRA
LoRA is based on the observation that the weight *update* needed to adapt a pre-trained model to a new task has a
low "intrinsic rank". For a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ (e.g. the query projection in an
attention layer) LoRA keeps $W_0$ frozen and learns the update as the product of two thin matrices:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k).$$

The forward pass becomes $h = W_0 x + \frac{\alpha}{r} B A x$. The key hyperparameters are:

- the **rank** $r$ (typically 8-64): the capacity of the update. The number of trainable parameters per matrix
  drops from $d \cdot k$ to $r (d + k)$; for $d = k = 4096$ and $r = 16$ that is 131k instead of 16.8M (0.8%).
- the **scaling** $\alpha$ (`lora_alpha`): the update is multiplied by $\alpha / r$, so that changing $r$ does not
  require retuning the learning rate. Common choices are $\alpha = r$ or $\alpha = 2r$.
- the **target modules**: which linear layers get adapters. Adapting all attention projections
  (`q_proj, k_proj, v_proj, o_proj`) and the MLP (`gate_proj, up_proj, down_proj`) usually works best.

$A$ is initialised randomly and $B$ with zeros, so at the start of training $\Delta W = 0$ and the adapted model is
exactly the pre-trained model. After training, $BA$ can be merged into $W_0$, so LoRA adds no inference latency,
or the small adapter files (a few MB to a few hundred MB) can be kept separately and swapped per task.

#### QLoRA
QLoRA (Dettmers et al. 2023) goes one step further and stores the frozen base weights $W_0$ in 4-bit precision
(the NormalFloat4 data type, plus "double quantisation" of the quantisation constants and paged optimisers to
survive memory spikes). The weights are de-quantised on the fly in the forward pass, while the LoRA matrices $A$ and
$B$ are kept and trained in 16-bit. Gradients flow through the frozen quantised model into the adapters. This
cuts the memory needed dramatically: full 16-bit fine-tuning of a 7B model with Adam needs well over 100 GB of GPU
memory, QLoRA fits into roughly 6-10 GB, and the QLoRA paper fine-tunes a 65B model on a single 48 GB GPU with
little loss of quality compared with 16-bit fine-tuning.

#### When to use PEFT
- Use LoRA/QLoRA when GPU memory is limited, when many task-specific variants of one base model are needed, or when
  the data set is small to medium sized (it also acts as a regulariser against catastrophic forgetting).
- Full fine-tuning can still be better when the new task is far from the pre-training distribution (e.g. a new
  language or domain with lots of data) and the budget allows it.
- PEFT is the default choice for both supervised fine-tuning and reinforcement learning of LLMs on modest hardware.

#### Example: supervised fine-tuning of GPT OSS with Unsloth
`s00_peft_unsloth/main_gpt_oss_peft_unsloth.py` loads GPT OSS 20B in 4-bit (by default the pre-quantised
checkpoint `unsloth/gpt-oss-20b-unsloth-bnb-4bit`, which downloads faster than `unsloth/gpt-oss-20b`) with
`FastLanguageModel.from_pretrained(..., load_in_4bit=True)`, adds LoRA adapters with
`FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16, target_modules=[...])` and trains them with TRL's
`SFTTrainer` on a finance question-answer data set. With `train_on_responses_only` the loss is computed only on the
assistant answers, not on the prompts. [Unsloth](https://docs.unsloth.ai) provides hand-written GPU kernels and
memory optimisations that make LoRA/QLoRA training noticeably faster and lighter than the plain HuggingFace stack.

### Reinforcement learning
Supervised fine-tuning (SFT) teaches a model to imitate reference answers. For many goals, however, it is easier to
*judge* an answer than to write the ideal one: is the answer helpful and harmless, is the final number correct,
does the code pass the unit tests? Reinforcement learning (RL) lets the model generate its own answers and adjusts
the weights so that answers with a high reward become more likely. In RL language the LLM is the **policy**
$\pi_\theta$, the prompt $q$ is the state, generating the completion $o$ token by token is a sequence of actions,
and a reward $r(q, o)$ is received at the end.

#### RLHF: reward model + PPO
Reinforcement learning from human feedback (RLHF), popularised by InstructGPT (Ouyang et al. 2022) and ChatGPT,
has three stages:

1. **SFT** on human-written demonstrations.
2. **Reward model**: humans rank several answers to the same prompt; a model $r_\phi(q, o)$ (usually the SFT model
   with a scalar head) is trained on pairs of preferred $o_w$ and rejected $o_l$ answers with the Bradley-Terry loss
   $-\log \sigma\left(r_\phi(q, o_w) - r_\phi(q, o_l)\right)$.
3. **RL with PPO** (Proximal Policy Optimisation, Schulman et al. 2017): the policy maximises the reward model's
   score while a KL penalty keeps it close to the SFT (reference) model, so it does not drift into gibberish that
   happens to fool the reward model:
   $$\max_\theta\; \mathbb{E}_{q,\, o \sim \pi_\theta}\left[ r_\phi(q, o) \right] - \beta\, \mathrm{KL}\left(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right).$$

PPO optimises a clipped surrogate objective. With the probability ratio
$\rho_t = \pi_\theta(o_t \mid q, o_{<t}) / \pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})$ and an advantage estimate
$\hat{A}_t$ (how much better the action was than expected) it maximises
$\min\left(\rho_t \hat{A}_t,\; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, \hat{A}_t\right)$, which prevents
overly large policy updates. The advantages are computed with a learned **value model** (critic) of the same size
as the policy. PPO-based RLHF therefore juggles four large models (policy, reference, reward and value model) and
is notoriously expensive and finicky to tune.

#### DPO
Direct Preference Optimisation (Rafailov et al. 2023) shows that the KL-regularised RLHF objective has a closed-form
optimal policy, which allows the reward model to be expressed through the policy itself. The result is a simple
classification-like loss on preference pairs, with no sampling, no reward model and no value model:
$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\left( \beta \log \frac{\pi_\theta(o_w \mid q)}{\pi_{\mathrm{ref}}(o_w \mid q)} - \beta \log \frac{\pi_\theta(o_l \mid q)}{\pi_{\mathrm{ref}}(o_l \mid q)} \right).$$
DPO is cheap and stable and is widely used for aligning chat models to preferences. As an *offline* method it only
learns from a fixed set of answers, though, and cannot explore new solutions the way online RL can.

#### GRPO
Group Relative Policy Optimisation (GRPO) was introduced in DeepSeekMath (Shao et al. 2024) and became famous as
the algorithm behind DeepSeek-R1 (DeepSeek-AI 2025). Its key idea is to drop the value model: for every prompt $q$
the policy samples a **group** of $G$ completions $o_1, \dots, o_G$, each is scored, and the advantage of each
completion is simply its reward standardised within the group:
$$\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}.$$
Every token of completion $o_i$ receives the advantage $\hat{A}_i$. Answers that are better than their siblings are
reinforced, worse ones are suppressed. The objective keeps PPO's clipping and adds the KL penalty directly to the
loss (rather than to the reward):
$$\mathcal{J}_{\mathrm{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\left[ \min\left(\rho_{i,t} \hat{A}_i,\; \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\, \hat{A}_i\right) -
\beta\, \mathbb{D}_{\mathrm{KL}}\left(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right) \right].$$
Consequences worth noting:

- No critic: only the policy and (if $\beta > 0$) a frozen reference model are needed. With LoRA the reference is
  just the base model with the adapters switched off, so it costs no extra memory.
- The learning signal comes from *differences within a group*. If all $G$ answers get the same reward (all right or
  all wrong) the advantages are zero and the prompt teaches nothing, so the task must be of the right difficulty and
  the sampling temperature high enough to produce diverse answers.
- Generation dominates the run time; libraries therefore offer fast inference back-ends such as vLLM.
- Later work refined the loss normalisation (e.g. Dr. GRPO and DAPO remove a bias towards long wrong answers); TRL
  exposes these variants via `loss_type`.

#### Verifiable rewards (RLVR)
For tasks with an objectively checkable result the reward model can be replaced by a program. This is called
reinforcement learning with verifiable rewards (RLVR): maths answers are compared with the reference number, code is
run against unit tests, and output formats are checked with regular expressions. DeepSeek-R1-Zero was trained with
GRPO from a base model using only such rule-based accuracy and format rewards and spontaneously learned long
chain-of-thought reasoning, self-verification and "aha moments". Rule-based rewards are cheap, deterministic and
much harder to exploit than a learned reward model.

#### Reward hacking
The policy optimises *exactly* what the reward measures, which is not necessarily what we meant (Goodhart's law).
Typical examples: a learned reward model that prefers long answers leads to verbose rambling; a format reward that
only checks for tags leads to empty `<reasoning></reasoning>` blocks; a code reward based on "tests pass" leads to
special-casing the tests. Counter-measures are the KL penalty, verifiable rewards wherever possible, combining
several rewards where the task reward (correctness) dominates the shaping rewards (format), capping or penalising
length, and — most importantly — regularly reading samples of the generated completions instead of only watching
the reward curve.

#### Example: GRPO on GSM8K with TRL and Unsloth
`s01_rl_unsloth/main_grpo_unsloth.py` (needs a GPU) teaches `unsloth/Qwen2.5-3B-Instruct` to solve grade-school
maths word problems (GSM8K) in a fixed format:

```
<reasoning>
...
</reasoning>
<answer>
42
</answer>
```

The walkthrough:

1. **Model**: the model is loaded in 4-bit with `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` and LoRA
   adapters ($r = 16$, all attention and MLP projections) are added with `get_peft_model` — i.e. QLoRA.
2. **Data**: each GSM8K row is mapped by `format_gsm8k_example` to a conversational `prompt` (system prompt
   describing the format + question) and the reference `answer`, extracted from the `#### 72` line of the solution.
3. **Rewards** (`utils/rewards.py`): TRL calls every reward function as `reward(prompts, completions, answer=...)`,
   where extra dataset columns arrive as keyword arguments, and expects one float per completion. The total reward
   is the sum of
   - `correctness_reward_func`: 2.0 if the number in `<answer>` equals the reference (the verifiable task reward),
   - `int_reward_func`: 0.5 if the answer is a plain integer,
   - `strict_format_reward_func` / `soft_format_reward_func`: 0.5 each for following the template exactly / loosely,
   - `xmlcount_reward_func`: 0.125 per correctly placed tag, minus a small penalty (0.001 per character) for text
     after the final `</answer>`; a trailing newline or a long reasoning is not penalised.

   The small shaping rewards give a learning signal early on, when hardly any answer is both correct and well
   formatted; correctness dominates so the format cannot be "hacked" at the expense of the task. The functions are
   pure Python and are unit-tested on CPU in `test_rewards.py`.
4. **Before/after evaluation**: the same 20 test questions are answered greedily before and after training, and
   accuracy and format compliance are reported. Because $B = 0$ at initialisation, the "before" model is the base model.
5. **Training**: `GRPOTrainer` with `GRPOConfig(num_generations=4, max_prompt_length=256, max_completion_length=768,
   beta=0.04, loss_type="bnpo", learning_rate=5e-6, max_steps=250, use_vllm=False)`. Each step samples 4
   completions for one prompt, scores them, computes group-relative advantages and updates the adapters with the
   clipped objective plus the KL penalty ($\beta = 0.04$) shown above. `loss_type="bnpo"` averages the loss over all
   completion tokens of the batch instead of per completion ($1/|o_i|$), which removes the length bias mentioned
   above. TRL's default `"dapo"` would normalise the same way here, but Unsloth then also switches on the other DAPO
   settings (asymmetric clipping with $\epsilon_{\text{high}} = 0.28$, masking of truncated completions) and
   recommends dropping the KL term ($\beta = 0$), so the script sets the loss type explicitly. The logged `reward` and
   `rewards/<function>/mean` columns show the format rewards rising first and correctness later.
6. **Saving**: only the LoRA adapters (plus tokeniser and metrics) are saved to `outputs/grpo_lora_<timestamp>`;
   setting `training = False` reloads them for inference.

The instruct model already solves many GSM8K problems, so the largest change is usually in format compliance, with
a modest accuracy gain after a few hundred steps. Longer runs, a larger group size, and `fast_inference=True` with
vLLM (Unsloth's recommended setup) give stronger results.

#### Unsloth docker images:
[See here](https://unsloth.ai/docs/get-started/install/docker)
