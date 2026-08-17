# E1–E12 GitHub 发布日志

日期：2026-08-17（Asia/Singapore）

## 上传范围

本次提交包含 E1–E12 可复现闭包：

- Flow Matching、x0 预测、Global loss weighting 和 Global/Local 双分支的核心代码；
- B0 与 E1–E12 的 YAML 配置；
- 训练、测评、论文口径指标、确定性配对 bootstrap 和推理测速脚本；
- Flow、消融和 Global/Local 输出头测试；
- `README.md`、`ablation/README.md`、`result.md` 和本日志。

明确不上传：

- `exp/`：约 540 GB，包含 checkpoint、TensorBoard、预测、指标和运行日志；
- `body_models/`：约 393 MB，SMPL-X 模型受单独许可约束；
- `__pycache__/`、`.pytest_cache/`、`*.pyc`、`*.orig`、checkpoint 和 TensorBoard 产物；
- 空的 `TODO.md`和未被实验文档引用的 `UniEgoMotion_坐标变换公式.docx`。

提交前已将新增的 `/gaozt-test1/...` 与 `/root/miniconda3/...` 本机路径改为仓库相对路径、`UEM_DATA_DIR` 或 `PYTHON_BIN`。这些修改只影响路径解析，不改变已测评模型的数值逻辑。

## 审计命令

```bash
git status --short --branch
git remote -v
git branch -vv
git log -5 --oneline --decorate
git diff --stat
git diff --check
find . -path './.git' -prune -o -path './exp' -prune -o -type f -printf '%s\t%p\n' | sort -nr | head -n 80
rg -n '/gaozt-test1|/root/miniconda3|guanzerong' config ablation README.md result.md
rg -n -i '(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|password\s*[:=]|token\s*[:=])' \
  .gitignore README.md config dataset eval model module mydiffusion run ablation tests result.md
curl -L -sS -o /dev/null -w '%{http_code}\n' https://github.com/sxh-kk/UniEgoMotion
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com
```

审计结果：

- 未发现 PAT、私钥、密码或 `.env` 凭据；
- 扫描中命中的 `mask_token`/`replace_token` 均为模型变量名，不是认证 Token；
- 待提交源码和文档无 GitHub 大文件风险；
- 原 `origin` 是 `chaitanya100100/UniEgoMotion`，只作为上游，不向其推送；
- `sxh-kk/UniEgoMotion` 公开页面在审计时返回 HTTP 404，当前环境也没有可用的 GitHub HTTPS/SSH 写入凭据。

## 验证命令

Shell 脚本语法：

```bash
for f in ablation/scripts/*.sh; do bash -n "$f"; done
```

结果：通过。

配对统计复算：

```bash
/root/miniconda3/envs/uem/bin/python ablation/scripts/paired_bootstrap_metrics.py \
  --experiment E7 exp/ablation/e7_x0_global_w8_u84k _ablation_e7_x0_global_w8_euler10 \
  --experiment E9 exp/ablation/e9_dual_no_fusion_w8_u84k _ablation_e9_dual_no_fusion_w8_euler10 \
  --experiment E10 exp/ablation/e10_dual_l2g_stopgrad_w8_u84k _ablation_e10_dual_l2g_stopgrad_w8_euler10 \
  --experiment E11 exp/ablation/e11_dual_g2l_stopgrad_w8_u84k _ablation_e11_dual_g2l_stopgrad_w8_euler10 \
  --experiment E12 exp/ablation/e12_dual_bidir_gated_w8_u84k _ablation_e12_dual_bidir_gated_w8_euler10 \
  --compare E7 E9 --compare E7 E10 --compare E7 E11 --compare E7 E12 \
  --compare E9 E10 --compare E9 E11 --compare E9 E12 \
  --bootstrap-resamples 10000 --bootstrap-seed 62 \
  --output /tmp/e7_e12_bootstrap.json
```

结果：通过。63 组“任务 × 指标 × 模型比较”的均值和 95% CI 与 `result.md` 未舍入统计一致；E7/E9–E12 的三任务有序预测键哈希一致，均为 `4155a5be70aeb19900651c66433f1cacb91500a5b38545ef9189c97883c87fca`。

B0–E8 的第一、二轮比较使用同一实现复算：

```bash
/root/miniconda3/envs/uem/bin/python ablation/scripts/paired_bootstrap_metrics.py \
  --experiment B0 exp/uem_flow_8gpu_e300 _report_b0_euler10_1gpu \
  --experiment E1 exp/uem_flow_8gpu_e300 _report_e1_euler50_1gpu \
  --experiment E2 exp/ablation/e2_global_w8_u84k _ablation_e2_global_w8_euler10 \
  --experiment E3 exp/ablation/e3_velocity_u168k _ablation_e3_u168k_euler10 \
  --experiment E4 exp/ablation/e4_x0_u84k _ablation_e4_x0_euler10 \
  --experiment E5 exp/ablation/e5_x0_global_w2_u84k _ablation_e5_x0_global_w2_euler10 \
  --experiment E6 exp/ablation/e6_x0_global_w4_u84k _ablation_e6_x0_global_w4_euler10 \
  --experiment E7 exp/ablation/e7_x0_global_w8_u84k _ablation_e7_x0_global_w8_euler10 \
  --experiment E8 exp/ablation/e8_x0_rot6_trans12_u84k _ablation_e8_x0_rot6_trans12_euler10 \
  --compare B0 E1 --compare B0 E2 --compare B0 E3 --compare B0 E4 \
  --compare E4 E5 --compare E4 E6 --compare E4 E7 --compare E4 E8 \
  --bootstrap-resamples 10000 --bootstrap-seed 62 \
  --output /tmp/b0_e8_bootstrap.json
```

结果：通过。72 组统计已用确定性实现复算，`result.md` 中引用的区间按该输出更新；B0–E8 的有序预测键哈希也全部一致。

YACS 配置解析与相对路径检查：

```bash
env -u UEM_DATA_DIR PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/uem/bin/python - <<'PY'
from pathlib import Path
from config.defaults import get_cfg_defaults

paths = [Path("config/uem_flow.yaml"), *sorted(Path("ablation/configs").glob("*.yaml"))]
for path in paths:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(path))
    assert not Path(str(cfg.DATA.DATA_DIR)).is_absolute(), (path, cfg.DATA.DATA_DIR)
    if cfg.TRAIN.EXP_PATH is not None:
        assert not Path(str(cfg.TRAIN.EXP_PATH)).is_absolute(), (path, cfg.TRAIN.EXP_PATH)
    if path.parent.name == "configs":
        assert cfg.EVAL.NUM_GPUS == 1, (path, cfg.EVAL.NUM_GPUS)
print(f"config merge passed: {len(paths)} files; all ablation evals are single-GPU")
PY
```

结果：`config merge passed: 14 files; all ablation evals are single-GPU`。

`pytest` 入口尝试：

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/uem/bin/python -m pytest -p no:cacheprovider -q \
  tests/test_flow_matching.py tests/test_flow_ablation.py tests/test_global_local_output.py
```

结果：当前训练环境未安装 `pytest`（`No module named pytest`），因此不改变依赖环境，改为直接执行测试脚本。首次未设 `PYTHONPATH` 时出现本地包导入失败，修正后使用：

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/uem/bin/python tests/test_flow_matching.py
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/uem/bin/python tests/test_flow_ablation.py
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/uem/bin/python tests/test_global_local_output.py
```

结果：三组全部通过，分别输出 `Flow Matching tests passed`、`Flow ablation tests passed`和 `Global/Local output head tests passed`。E9–E12 已各完成 300 epoch 真实双卡 DDP 训练，因本次发布整理未改变模型/DDP 逻辑，未重复运行 smoke test。

额外语法检查：

```bash
/root/miniconda3/envs/uem/bin/python -m py_compile \
  mydiffusion/flow_matching.py ablation/scripts/paired_bootstrap_metrics.py
```

结果：通过。

## Git 提交与推送命令

```bash
git config user.name sxh-kk
git config user.email 1554334352@qq.com
git remote add sxh https://github.com/sxh-kk/UniEgoMotion.git
git add .gitignore README.md LOG.md result.md \
  config/defaults.py config/uem_flow.yaml \
  dataset/ee4d_motion_dataset.py dataset/smpl_utils.py \
  eval/compute_3d_metrics.py eval/eval_exp.py eval/save_uem_preds.py \
  model/uniegomotion.py module/ema.py module/uem_module.py \
  mydiffusion/flow_matching.py run/train_uem.py \
  ablation/README.md ablation/configs/*.yaml ablation/scripts/*.py ablation/scripts/*.sh \
  tests/*.py
git diff --cached --check
git diff --cached --name-status
git commit -m 'E1-E12'
git show -s --format=fuller HEAD
```

本地提交成功，提交主题为 `E1-E12`，Author 和 Committer 均为 `sxh-kk <1554334352@qq.com>`。首次非交互 dry-run 的 HTTPS 请求长时间未返回，终止后用 30 秒上限和禁用 askpass 的方式复核：

```bash
timeout 30s env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false \
  git push --dry-run --no-verify sxh HEAD:refs/heads/main
```

结果：失败，退出码 128；Git 报告无法读取 GitHub 用户名且终端提示已禁用。这与提交前的认证审计一致，说明当前环境没有可用的 GitHub HTTPS 写入凭据。由于 dry-run 未成功，以下正式 push **没有执行**：

```bash
GIT_TERMINAL_PROMPT=0 git push sxh HEAD:refs/heads/main
```

因此当前状态是“本地提交完成、GitHub 上传等待仓库和认证”。配置有效凭据且确认 `sxh-kk/UniEgoMotion` 已创建后，应重新运行 dry-run，成功后才能执行正式 push。不使用裸 `git push`或 `git add .`，不使用 force push，不将 PAT 写入 remote URL 或本日志。

## 重新上传重试

用户再次要求上传后，重新核对了提交与两个远端，并再次执行非交互 dry-run：

```bash
git status --short --branch
git show -s --format='%H %s%n%an <%ae>' HEAD
git remote get-url sxh
git remote get-url origin
timeout 60s env GIT_TERMINAL_PROMPT=0 \
  git push --dry-run --no-verify sxh HEAD:refs/heads/main
timeout 60s env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false \
  git push --dry-run --no-verify sxh HEAD:refs/heads/main
timeout 20s ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  -T git@github.com </dev/null
curl --max-time 20 -L -sS -o /dev/null -w '%{http_code}\n' \
  https://github.com/sxh-kk/UniEgoMotion
```

结果：第一次 dry-run 在 60 秒上限内未通过凭据交互；禁用 askpass 后以退出码 128 明确失败，仍提示无法读取 GitHub 用户名。SSH 返回 `Permission denied (publickey)`，目标公开页面返回 HTTP 404。正式 push 再次没有执行，`origin` 未被写入。需要先创建目标仓库（若尚未创建）并在当前环境配置 sxh-kk 的 HTTPS Token/credential helper 或已注册的 SSH key。

## 目标仓库更正

用户随后明确提供目标仓库 `https://github.com/sxh-kk/ExpectionToSmplx.git`。执行以下命令核对并更新远端：

```bash
timeout 30s env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false \
  git ls-remote --heads --tags https://github.com/sxh-kk/ExpectionToSmplx.git
git remote set-url sxh https://github.com/sxh-kk/ExpectionToSmplx.git
timeout 60s env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false \
  git fetch --no-tags sxh main
git show -s --format=fuller sxh/main
git ls-tree -r --name-only sxh/main
git merge-base HEAD sxh/main
git show sxh/main:README.md
```

结果：远端 `main` 可公开读取，只有提交 `e3bb229`（`Initial commit`）和内容为 `# ExpectionToSmplx` 的 `README.md`；它与 UniEgoMotion 历史没有共同祖先。为保留该初始提交、避免 force push，同时以本项目完整 README 和代码为最终工作树，使用可恢复的 `ours` merge 连接两段历史：

```bash
git merge --allow-unrelated-histories -s ours sxh/main \
  -m 'Merge ExpectionToSmplx repository history'
```

远端原始 README 仍完整保存在 `e3bb229` 中。合并后只允许显式向 `sxh` 的 `main` 做普通 push；如果服务器拒绝认证或分支策略检查，则停止且不改写远端历史。

合并命令执行成功，`sxh/main` 已成为本地 `main` 的祖先。随后执行三次非写入 dry-run：

```bash
timeout 60s env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false \
  git push --dry-run --no-verify sxh HEAD:refs/heads/main
timeout 60s env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false \
  git -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 \
  push --dry-run --no-verify sxh HEAD:refs/heads/main
timeout 60s env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false \
  git push --dry-run --no-verify \
  https://sxh-kk@github.com/sxh-kk/ExpectionToSmplx.git \
  HEAD:refs/heads/main
```

结果：第一次遇到瞬时 TLS 连接终止；第二次到达认证阶段后提示无法读取 GitHub 用户名；第三次显式给出用户名后提示无法读取密码/Token。当前环境仍无目标仓库的写入凭据，所以正式 push 没有执行。远端 `main` 仍停留在 `e3bb229`，没有发生部分上传或历史覆盖。

## SSH deploy key 配置

为避免在命令、remote URL 或日志中存放 GitHub Token，改用仅授权给 `ExpectionToSmplx` 的独立 SSH deploy key。现有 `ssh_worker_rsa_key` 经显式测试未被 GitHub 接受，因此未复用、未修改：

```bash
ssh-keygen -lf /root/.ssh/ssh_worker_rsa_key.pub
timeout 20s ssh -i /root/.ssh/ssh_worker_rsa_key \
  -o IdentitiesOnly=yes -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new -T git@github.com </dev/null
```

专用 key 的生成与本地配置命令：

```bash
umask 077
ssh-keygen -q -t ed25519 \
  -C 'sxh-kk/ExpectionToSmplx deploy key' \
  -f /root/.ssh/github_sxh_kk_expectiontosmplx_ed25519 -N ''
chmod 600 /root/.ssh/github_sxh_kk_expectiontosmplx_ed25519
chmod 644 /root/.ssh/github_sxh_kk_expectiontosmplx_ed25519.pub
ssh-keygen -lf /root/.ssh/github_sxh_kk_expectiontosmplx_ed25519.pub
chmod 600 /root/.ssh/config
git remote set-url sxh \
  git@github-sxh-kk-expectiontosmplx:sxh-kk/ExpectionToSmplx.git
```

`/root/.ssh/config` 中的专用 Host alias 固定使用该私钥并启用 `IdentitiesOnly yes`。私钥权限为 600，未输出、未加入 Git；公钥指纹为 `SHA256:V1gFGvWkolFuMP46HItZjecHiYT4ewQ+I5ev1fMmOlk`。下一步需要仓库管理员在 GitHub 的 `Settings → Deploy keys` 中添加对应公钥并勾选 `Allow write access`，之后再测试 SSH 和执行 push。

## SSH 授权与最终上传

仓库管理员添加 deploy key 并授予写权限后，执行：

```bash
timeout 20s ssh -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  -T github-sxh-kk-expectiontosmplx </dev/null
timeout 60s env GIT_TERMINAL_PROMPT=0 \
  git push --dry-run --no-verify sxh HEAD:refs/heads/main
```

结果：SSH 返回 GitHub 的成功认证提示并识别为 `sxh-kk/ExpectionToSmplx`；退出码 1 是 GitHub 不提供交互式 shell 的预期行为。dry-run 退出码为 0，确认 `main` 可从远端 `e3bb229` 普通快进到本地提交。日志并入最终 merge commit 后，重新 dry-run 并执行：

```bash
GIT_TERMINAL_PROMPT=0 git push sxh HEAD:refs/heads/main
```

该命令只更新 `sxh/ExpectionToSmplx` 的 `main`，不操作上游 `origin`，不使用 force push。
