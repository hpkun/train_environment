# Experiment guide

## 0. Paper environment defaults

The vanilla MAPPO entry point now defaults to the paper environment scale:
6v6, `max_episode_length=1400`, `obs_mode="paper_strict"`, and
`enable_gcas_for_blue=False`. This is still a pure MAPPO baseline, not the
BRMA algorithm.

Use the paper main preset for full-scale environment reproduction:

```powershell
python train_vanilla_mappo.py --preset vanilla_6v6_paper_main
```

Use 1v1/2v2 presets only as smoke or debugging runs. For example:

```powershell
python train_vanilla_mappo.py --preset vanilla_2v2_smoke --total-env-steps 2000
```

Current reward version is `paper_literal_eq15_eq20_ta1_tail01_joint_v4`:
Eq.20 uses `Ta=1.0` below 4 degrees and Eq.17 uses the paper-explicit `0.1`
tail. Situation reward currently uses observer velocity-to-LOS q_LOS; this
geometry remains `UNRESOLVED / PAPER_INFERRED` rather than a confirmed match.

This is a 3V3 Vanilla MAPPO diagnostic, not the paper's complete 6V6
BRMA-MAPPO numerical reproduction. The paper table records `1.5e7` maximum
training steps. Altitude thresholds, `D_att,max`, PID gains, initial state,
formation spacing, missile count, RCS/radar constants, EO half-angle, MWS,
Blue policy details, action distribution/std, and full missile dynamics remain
paper-unspecified engineering choices.

## 1. Conda 鐜

鏈」鐩富瑕佸湪 `brmamappo` conda 鐜涓繍琛屻€俉indows PowerShell 绀轰緥锛?

```powershell
conda activate D:\conda_envs\envs_dirs\brmamappo
```

涔熷彲浠ョ洿鎺ヤ娇鐢ㄨ鐜鐨?Python 瑙ｉ噴鍣細

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe ...
```

## 2. 蹇€?smoke test

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe train_vanilla_mappo.py --num-red 1 --num-blue 1 --num-envs 1 --total-env-steps 20 --replay-buffer-size 10 --max-episode-length 10 --device cpu --log-file smoke_train_log.csv --results-file results/smoke_results.csv --checkpoint-dir smoke_checkpoints
```

璇ュ懡浠ゅ彧鐢ㄤ簬楠岃瘉璁粌閾捐矾鏄惁鑳藉惎鍔ㄣ€佸啓鏃ュ織鍜屼繚瀛樼粨鏋滐紝涓嶄唬琛ㄦ湁鏁堣缁冪粨鏋溿€?

## 3. 褰撳墠榛樿璁粌鍛戒护

榛樿浠嶆槸 2v2 vanilla MAPPO baseline锛?

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe train_vanilla_mappo.py
```

榛樿杈撳嚭锛?

- `vanilla_training_log.csv`
- `results/vanilla_mappo_results.csv`
- `checkpoints/`
  - `vanilla_actor_latest_*.pt` / `centralized_critic_latest_*.pt` 鈥?姣?10 iter 杞浆 (淇濈暀鏈€鏂?5 涓?
  - `vanilla_actor_best_reward.pt` / `centralized_critic_best_reward.pt` 鈥?杩戞湡骞冲潎濂栧姳鏈€楂?
  - `vanilla_actor_best_winrate.pt` / `centralized_critic_best_winrate.pt` 鈥?杩戞湡鑳滅巼鏈€楂?(reward tie-breaker)
  - `vanilla_actor_best.pt` / `centralized_critic_best.pt` 鈥?best_winrate 鐨勫吋瀹瑰埆鍚?
  - `vanilla_actor_final.pt` / `centralized_critic_final.pt` 鈥?璁粌缁撴潫鏈€缁堟ā鍨?

璁粌榛樿 `enable_blue_gcas=False`銆?

## 4. Preset-based commands

椤圭洰鏀寔閫氳繃 `--preset` 缂╃煭甯哥敤鍛戒护銆傚垪鍑烘墍鏈?preset锛?

```powershell
conda activate brmamappo
python train_vanilla_mappo.py --list-presets
python train_attention_mappo.py --list-presets
```

甯哥敤 preset 绀轰緥锛?

```powershell
# vanilla 1v1 smoke (20 steps, cpu)
python train_vanilla_mappo.py --preset vanilla_1v1_smoke

# vanilla 2v2 smoke (~10k steps, see reward signals)
python train_vanilla_mappo.py --preset vanilla_2v2_smoke

# vanilla 6v6 paper main (10M steps, full training)
python train_vanilla_mappo.py --preset vanilla_6v6_paper_main

# attention smoke variants
python train_attention_mappo.py --preset attention_1v1_smoke
python train_attention_mappo.py --preset attention_2v2_current_smoke
python train_attention_mappo.py --preset attention_2v2_placeholder_smoke
```

CLI 鍙傛暟浠嶅彲瑕嗙洊 preset锛?

```powershell
python train_vanilla_mappo.py --preset vanilla_2v2_smoke --total-env-steps 2000
```

Current default reward version: `paper_literal_eq15_eq20_ta1_tail01_joint_v4`. See:
[docs/current_environment_alignment_status.md](current_environment_alignment_status.md)銆?

## 5. 璁烘枃寮?6v6 璁粌鍛戒护妯℃澘

涓嬮潰鏄?6v6 璁粌鍛戒护妯℃澘銆傛敞鎰忥細褰撳墠浠嶅彧鏄?vanilla MAPPO baseline锛屼笉鏄?BRMA-MAPPO銆?

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe train_vanilla_mappo.py --num-red 6 --num-blue 6 --num-envs 8 --total-env-steps 10000000 --max-episode-length 1400 --device auto --log-file logs/vanilla_6v6.csv --results-file results/vanilla_6v6_results.csv --checkpoint-dir checkpoints_vanilla_6v6
```

## 6. 鎵归噺璇勪及

`evaluate_vanilla_mappo.py` 涓嶇敓鎴?ACMI锛屼富瑕佺敤浜庡灞€缁熻璁烘枃寮忔寚鏍囥€傞粯璁?`enable_blue_gcas=False`锛屼笌璁粌鑴氭湰鍜?ACMI 鍗曞眬璇勪及淇濇寔涓€鑷淬€傝嫢闇€瑕佹樉寮忓紑鍚摑鏂?GCAS锛屽彲娣诲姞 `--enable-blue-gcas`銆?

闅忔満绛栫暐 smoke test锛?

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe evaluate_vanilla_mappo.py --random --num-red 1 --num-blue 1 --episodes 2 --max-steps 10 --device cpu --output results/smoke_eval_metrics.csv
```

璇勪及 trained 2v2 checkpoint锛?

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe evaluate_vanilla_mappo.py --checkpoint checkpoints/vanilla_actor_best.pt --num-red 2 --num-blue 2 --episodes 20 --max-steps 1400 --device auto --output results/eval_2v2.csv
```

vanilla MLP baseline 鐨?flattened observation 缁村害闅忚妯″彉鍖栵紝鍥犳涓嶈兘鐩存帴鎶?2v2 checkpoint 鐢ㄥ埌 6v6銆?v8 鎴?10v10銆傝繖涓嶆槸 BRMA zero-shot 璁剧疆銆?

## 7. Tacview ACMI 鍗曞眬鍙鍖?

`eval_acmi.py` 鐢ㄤ簬鍗曞眬 Tacview 鍙鍖栵紝涓嶇敤浜庢壒閲忕粺璁°€傝鑴氭湰榛樿鏄惧紡浣跨敤 `enable_gcas_for_blue=False`銆?

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe eval_acmi.py --checkpoint checkpoints/vanilla_actor_best.pt --num-red 2 --num-blue 2 --max-steps 1400 --output eval_battle.acmi
```

闅忔満绛栫暐 smoke test锛?

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe eval_acmi.py --random --num-red 1 --num-blue 1 --max-steps 10 --output smoke_eval.acmi
```

## 8. 褰撳墠宸插榻愯鏂囩殑鍐呭

- 闆疯揪 `Rmax = K * RCS^(1/4)`銆?
- 瀵煎脊 `0.25s` lock delay銆?
- 瀵煎脊 `0.5s` launch interval銆?
- 瀵煎脊鍛戒腑姒傜巼浣跨敤 missile velocity 涓?LOS dot product銆?
- boundary reward 浣跨敤 eq.18 鐨勫浐瀹氬崟娆¤秺鐣屾儵缃氥€?
- roll reward 浣跨敤 eq.16 double-condition銆?
- altitude reward 浣跨敤浜屾鍒嗘杩戜技銆?
- terminal reward 鎸?per-agent API 鍧囧垎銆?
- 澧炲姞璁烘枃寮忚瘎浼版寚鏍囥€?

## 9. 褰撳墠浠嶆湭瀵归綈璁烘枃鐨勫唴瀹?

- 榛樿璁粌浠嶆槸 2v2锛屼笉鏄鏂?6v6銆?
- 绠楁硶浠嶆槸 vanilla MAPPO锛屼笉鏄?BRMA-MAPPO銆?
- 灏氭湭瀹炵幇 EntityObservationEncoder銆?
- 灏氭湭瀹炵幇 MaskVectorGenerator銆?
- 灏氭湭瀹炵幇 biased random masked attention銆?
- observation 浠嶆槸褰撳墠 11 缁村伐绋嬪寲 entity vector锛屼笉鏄弗鏍?Table 1 / Table 2銆?
- critic 浠嶄娇鐢?red agents flattened observations concat锛屼笉鏄鏂?native global state銆?
- RCS 浠嶆槸 front/side approximation锛屼笉鏄鏂?RCS table interpolation銆?
- PID 鎺у埗鍣ㄥ惈宸ョ▼绋冲畾椤广€?
- 璁烘枃娌℃湁鏄庣‘缁欏嚭姣忔灦 UAV 鐨勫浐瀹氳浇寮归噺锛涘綋鍓嶇幆澧冧繚鐣欓粯璁?`num_missiles_per_plane=999`锛岀瓑浠蜂簬涓嶈杞藉脊閲忔垚涓轰富瑕侀檺鍒跺洜绱犮€傜敱浜庤鏂囨病鏈夋彁渚涘叿浣撹浇寮归噺锛岃椤规殏涓嶄綔涓轰紭鍏堝榻愮洰鏍囥€?

## 10. Git ignore 娉ㄦ剰浜嬮」

浠ヤ笅鏂囦欢涓嶅簲鎻愪氦锛?

- `smoke_train_log.csv`
- `smoke_checkpoints/`
- `smoke_eval.acmi`
- `results/smoke_*.csv`
- `__pycache__/`
- `*.pyc`

濡傛灉鐢熸垚浜嗕笂杩版枃浠讹紝璇蜂繚鎸佸畠浠浜?git ignored 鐘舵€侊紝涓嶈鍔犲叆鎻愪氦銆?

## 11. 涓嬩竴闃舵锛欵ntityObservationEncoder 鍑嗗

- 宸叉柊澧?`entity_obs_utils.py`锛屽彲灏嗗綋鍓?Dict observation 杞垚 entity-wise tensor銆?
- 褰撳墠 tensor 鏆傛椂浠嶄娇鐢ㄧ幆澧冪殑 11 缁村伐绋嬪寲 entity vector銆?
- 璇ュ伐鍏锋殏鏈帴鍏ヨ缁冿紝鍙敤浜庡悗缁疄鐜?MAPPO-Attention / BRMA-MAPPO銆?
- 鍚庣画浠嶉渶鍐冲畾鏄惁涓ユ牸鏀规垚璁烘枃 Table 1 / Table 2 鐨?10 缁磋〃绀恒€?

## 12. MAPPO-Attention 鍑嗗

- 宸叉柊澧?`attention_models.py`銆?
- 鐩墠鍖呭惈 `EntityObservationEncoder`銆乣AttentionActor`銆乣AttentionCritic`銆?
- 褰撳墠妯″潡灏氭湭鎺ュ叆璁粌锛屼粎閫氳繃绾?PyTorch smoke test 楠岃瘉 shape銆?
- 涓嬩竴姝ユ墠浼氭柊澧?`train_attention_mappo.py` 鎴栧湪鐙珛鍒嗘敮涓帴鍏ヨ缁冦€?
- 褰撳墠 attention encoder 浣跨敤 11 缁村伐绋嬪寲 entity vector锛屼笉鏄渶缁堣鏂?Table 1 / Table 2 鐨?10 缁翠弗鏍肩増鏈€?
- 褰撳墠杩樻病鏈夊疄鐜?biased random mask 鍜?mask vector generator銆?

## 13. MAPPO-Attention baseline

- 宸叉柊澧?`train_attention_mappo.py`銆?
- 杩欐槸 actor-side EntityObservationEncoder baseline銆?
- Critic 鏆傛椂浠嶄娇鐢?flattened red observations concat 鐨?centralized critic銆?
- 灏氭湭瀹炵幇 biased random mask 鍜?MaskVectorGenerator銆?
- 榛樿杈撳嚭锛?
  - `attention_training_log.csv`
  - `results/attention_mappo_results.csv`
  - `checkpoints_attention/`
- `train_attention_mappo.py` 鏀寔 `--obs-adapter current` 鍜?`--obs-adapter paper-placeholder`銆?
- `current` 鏄粯璁わ紝浣跨敤褰撳墠 11 缁村伐绋嬪寲 entity vector銆?
- `paper-placeholder` 浣跨敤 10 缁?placeholder adapter锛屼笉鏄?strict Table 1/Table 2 鐗╃悊閲忋€?
- strict paper extractor 宸插湪 `paper_state_extractor.py` 涓綔涓哄師鍨嬪瓨鍦紝浣嗗皻鏈帴鍏?SubprocVecEnv 璁粌銆?
- 浣跨敤 `paper-placeholder` 鏃跺簲浣跨敤鐙珛 checkpoint 鐩綍锛屼緥濡?`checkpoints_attention_paper_placeholder`銆?

smoke 鍛戒护锛?

```powershell
conda activate brmamappo
python train_attention_mappo.py --num-red 1 --num-blue 1 --num-envs 1 --total-env-steps 20 --replay-buffer-size 10 --max-episode-length 10 --device cpu --log-file smoke_attention_log.csv --results-file results/smoke_attention_results.csv --checkpoint-dir smoke_attention_checkpoints
```

杩欐潯鍛戒护浼氳Е鍙?JSBSim 鐜 reset锛孋odex 涓嶈繍琛岋紱鐢辨湰鍦扮敤鎴疯繍琛屻€?

paper-placeholder smoke 鍛戒护锛?

```powershell
conda activate brmamappo
python train_attention_mappo.py --obs-adapter paper-placeholder --num-red 1 --num-blue 1 --num-envs 1 --total-env-steps 20 --replay-buffer-size 10 --max-episode-length 10 --device cpu --log-file smoke_attention_paper_log.csv --results-file results/smoke_attention_paper_results.csv --checkpoint-dir smoke_attention_paper_checkpoints
```

杩欐潯鍛戒护鍚屾牱浼氳Е鍙?JSBSim 鐜 reset锛孋odex 涓嶈繍琛岋紱鐢辨湰鍦扮敤鎴疯繍琛屻€?

## 14. 璁烘枃寮?observation adapter 鍑嗗

- 宸叉柊澧?`paper_obs_utils.py`銆?
- 褰撳墠鍙槸鎶婄幇鏈?11 缁村伐绋嬪寲 entity vector 杞垚 10 缁存帴鍙ｅ崰浣嶃€?
- 瀹冭繕涓嶆槸涓ユ牸璁烘枃 Table 1/Table 2 鐨勭墿鐞嗛噺澶嶇幇銆?
- 鍚庣画鑻ヨ涓ユ牸澶嶇幇锛岄渶瑕佷粠 simulator/native state 涓瀯閫狅細
  - self state: `x, y, h, V, phi, theta, psi, alpha, beta, Vd`
  - relative state: `x_body, y_body, z_body, theta_v_body, psi_v_body, V, theta_LOS_body, psi_LOS_body, q_LOS, d`
- 鍦ㄥ畬鎴?strict observation 鍓嶏紝`train_attention_mappo.py` 鐨勭粨鏋滃彧鑳借浣滃伐绋?baseline锛岃€屼笉鏄鏂?MAPPO-Attention 娑堣瀺缁撴灉銆?

## 15. Strict paper observation prototype

- `UavCombatEnv` 宸叉毚闇?`get_strict_entity_observation(agent_id)` 鍜?
  `get_strict_team_observations(team)`銆?
- `reset()`/`step()` 榛樿 observation 浠嶆槸 11 缁村伐绋?Dict锛屼笉鍙楀奖鍝嶃€?
- strict API 鍚庣画鍙敤浜?`train_attention_mappo.py` 鐨?paper-strict adapter銆?

smoke 鍛戒护锛堣Е鍙?JSBSim锛孋odex 涓嶈繍琛岋紝鐢ㄦ埛鏈湴杩愯锛夛細

```powershell
conda activate brmamappo
python scripts/smoke_strict_observation_env.py
```

- `paper_state_extractor.py` 浠嶅湪 `my_uav_env/alignment/state_extractor.py` 涓€?
- 瀹冨皾璇曚粠 simulator/native state 鏋勯€犺鏂?Table 1/Table 2 鐨?10 缁磋娴嬨€?
- 褰撳墠 `alpha/beta` 鍙兘浠嶆槸 placeholder 0锛岄櫎闈?simulator 宸叉彁渚涘搴斿睘鎬с€?
- pass13 鍚?extractor 浼氬皾璇曚粠 JSBSim property 璇诲彇 `aero/alpha-rad`銆乣aero/alpha-deg`銆乣aero/beta-rad`銆乣aero/beta-deg`銆?
- extractor 鐜板湪浼氬湪 meta 涓褰?`alpha/beta` 鐨勬潵婧愩€?
- `q_LOS` 鐨勫畾涔変粛闇€鍜岃鏂囧嚑浣曞畾涔夋牳瀵广€?
- 褰撳墠 `q_LOS` 鏄?observer body x-axis angle placeholder锛屼笉绛夊悓浜?3-9 绾垮熬鍚庤銆?
- 鍚庣画鎺ュ叆璁粌鍓嶏紝杩橀渶瑕佽繘涓€姝ラ獙璇?`q_LOS` 涓庣幇鏈?AO/TA 鐨勫叧绯汇€?
- `radar_detected=False` 鏃朵細鎸夎鏂?Table 2 Note 灏嗛€熷害瑙掑拰鐩爣閫熷害缃?0銆?
- `scripts/smoke_paper_state_extractor_env.py` 浼氭墦鍗版瘡涓?entity 鐨勭墿鐞嗗瓧娈碉紝鐢ㄤ簬鏈湴妫€鏌ユ暟鍊兼柟鍚戝拰閲忕骇銆?
- Codex 涓嶈繍琛岃鑴氭湰锛岀敤鎴锋湰鍦拌繍琛屻€?
- 璇ユā鍧楀皻鏈帴鍏ヨ缁冿紱鍚庣画闇€瑕佸厛楠岃瘉鏁板€煎悎鐞嗘€э紝鍐嶅喅瀹氭槸鍚﹁ `train_attention_mappo.py` 浣跨敤瀹冦€?

## 16. MAPPO-Attention 鎵归噺璇勪及

- 宸叉柊澧?`evaluate_attention_mappo.py`銆?
- 瀹冭瘎浼?attention actor checkpoint锛屼笉鐢熸垚 ACMI銆?
- 鏀寔 `--obs-adapter current` / `--obs-adapter paper-placeholder`銆?
- checkpoint 鐨?`entity_dim` 蹇呴』鍜?`obs_adapter` 鍖归厤銆?
- 浠嶆湭瀹炵幇 BRMA mask銆?

current adapter 璇勪及妯℃澘锛?

```powershell
conda activate brmamappo
python evaluate_attention_mappo.py --checkpoint checkpoints_attention/attention_actor_best.pt --obs-adapter current --num-red 2 --num-blue 2 --episodes 20 --max-steps 1400 --device auto --output results/eval_attention_2v2.csv
```

paper-placeholder adapter 璇勪及妯℃澘锛?

```powershell
conda activate brmamappo
python evaluate_attention_mappo.py --checkpoint checkpoints_attention_paper_placeholder/attention_actor_best.pt --obs-adapter paper-placeholder --num-red 2 --num-blue 2 --episodes 20 --max-steps 1400 --device auto --output results/eval_attention_paper_placeholder_2v2.csv
```

闅忔満 smoke 绀轰緥锛?

```powershell
conda activate brmamappo
python evaluate_attention_mappo.py --random --obs-adapter current --num-red 1 --num-blue 1 --episodes 2 --max-steps 10 --device cpu --output results/smoke_eval_attention.csv
```

杩欐潯 smoke 鍛戒护浼氳Е鍙?JSBSim 鐜 reset锛孋odex 涓嶈繍琛岋紱鐢辨湰鍦扮敤鎴疯繍琛屻€?

## 17. Paper-style critic global state candidate

`my_uav_env/alignment/global_state.py` 鎻愪緵 strict team global state flatten 宸ュ叿銆?
褰撳墠 attention critic 浠嶄娇鐢?`obs_dim * num_red`锛?v2 绾?106 缁达級鐨?engineering flatten銆?
strict candidate 2v2 缁村害涓?88锛? entities 脳 10 dim + 4 mask = 44 per agent 脳 2锛夈€?

鏈?pass 鍙仛鍊欓€夊伐鍏凤紝涓嶆敼鍙樿缁冭涓恒€傚悗缁皢鍗曠嫭鍋?critic switch pass銆?

`train_attention_mappo.py` 鐜板凡鏀寔 `--critic-state`锛?
- `--critic-state engineering`锛堥粯璁わ級锛歝ritic 浣跨敤 flattened 11-dim obs concat銆?
- `--critic-state strict-global`锛歝ritic 浣跨敤 strict team global state锛堥渶 `--obs-adapter strict`锛夈€?
  2v2 strict-global dim = 88锛宔ngineering dim = 106銆?

鏂?preset:

```powershell
python train_attention_mappo.py --preset attention_1v1_strict_critic_smoke
```

## 18. Reward version 鏍囪

Current reward version: `paper_literal_eq15_eq20_ta1_tail01_joint_v4`.

瀹屾暣鐨勭幆澧冨榻愮姸鎬佽 [docs/current_environment_alignment_status.md](current_environment_alignment_status.md)銆?

`paper_literal_eq15_eq20_ta1_tail01_joint_v4` means:

1. Eq.20 first branch uses the paper-explicit `Ta=1.0`; the other branches are preserved without smoothing.
2. altitude reward 浣跨敤 pairwise eq.17-style锛堝惈 high-altitude 0.1 tail锛夛紱
3. Situation reward uses 3D velocity-to-LOS q_LOS plus 3D distance. Geometry alignment is `UNRESOLVED / PAPER_INFERRED`.

娉ㄦ剰锛?

- 涓嶈涓?`fixed_ta_v1`銆乣fixed_ta_alt_eq17_v1`銆佹垨 legacy reward 鏃ュ織娣峰悎姣旇緝銆?
- 鏂板疄楠屽缓璁娇鐢ㄥ甫鐗堟湰鍚嶇殑鏃ュ織鏂囦欢锛屼緥濡?`vanilla_3dlos_v1.csv`銆?
- Do not directly compare older reward-version checkpoints or logs with `paper_literal_eq15_eq20_ta1_tail01_joint_v4`.
## 18. Attention strict observation adapter

`train_attention_mappo.py` now supports three actor observation adapters:

- `--obs-adapter current`: default 11-dim engineering entity vector.
- `--obs-adapter paper-placeholder`: 10-dim placeholder projection from current env obs.
- `--obs-adapter strict`: 10-dim strict Table 1/Table 2 prototype observation from `UavCombatEnv.get_strict_team_observations("red")`.

Strict mode only changes the attention actor input. It does not change `reset()` / `step()` default observation, does not change `UavCombatEnv.observation_space`, and does not change the centralized critic. The critic still uses flattened 11-dim red observations concat.

Strict smoke preset:

```powershell
conda activate brmamappo
python train_attention_mappo.py --preset attention_1v1_strict_smoke
```

This command triggers JSBSim/env reset and is for local user runs only; Codex does not run it.

## 19. Blue no-target cruise boundary patrol

Blue rule policy target pursuit is still based on its existing observation and
target-selection logic. It has not been given radar-blind red-position tracking.

When Blue has no valid target, the old cruise branch kept the current heading.
In Tacview this could look like Blue never turns back and eventually leaves the
battlefield. The training and evaluation entry points now pass Blue ownship
positions from `UavCombatEnv.get_blue_own_positions()` into the rule policy, so
no-target Blue cruise turns back toward the battlefield center near the
boundary.

This is rule-based battlefield keeping, not a learning observation and not
radar-blind Red tracking. The helper receives only Blue ownship positions, not
Red positions.

The patrol now starts before the 40 km boundary and lowers no-target cruise
speed as Blue approaches the edge. It is still not a hard boundary: crossing can
still happen, but straight-line exits should be reduced. When using
`--draw-boundary`, Tacview can show whether Blue begins turning before the edge.
The patrol is closed-loop on Blue's own outward motion: if Blue is flying
outward near the boundary it turns and slows more strongly; if it is already
turning inward, correction is reduced to avoid oscillation.
If Blue is very close to the boundary and still flying outward, the same
ownship-only safety layer can override combat pursuit heading/speed so pursuit
does not drag Blue out of the battlefield. This still does not expose Red global
position or enforce a hard boundary.
Tacview nose direction corresponds to the simulator yaw (`sim.get_rpy()[2]`).
Training and evaluation now pass that yaw into the Blue rule policy through
`get_blue_own_kinematics()`, so boundary safety uses yaw heading when available
instead of velocity-track heading. Velocity heading remains only a compatibility
fallback.

After a head-on merge, Blue should not immediately straight-cruise only because
the radar track is temporarily lost. The rule policy now distinguishes radar
tracks from AWACS coarse tracks in the existing observation, pursues AWACS
coarse bearings with lower confidence, and keeps a short last-bearing memory
for reacquisition. This uses only observation-space target bearing/range and
Blue ownship state; it does not read Red global position.

## 20. ACMI battlefield boundary debug

`eval_acmi.py` does not draw battlefield boundaries by default. For Tacview
debugging only, enable optional boundary corner markers:

```powershell
python eval_acmi.py --checkpoint checkpoints/vanilla_actor_best.pt --num-red 2 --num-blue 2 --max-steps 1400 --output eval_battle.acmi --draw-boundary --boundary-half-size 40000
```

Normal replay commands should omit `--draw-boundary`.

## 21. Missile launch diagnostics

`train_vanilla_mappo.py` logs additional missile launch diagnostic fields. They
are counters only; they do not change automatic firing, missile dynamics,
radar detection, reward, or PPO training.

Key fields:

- `LaunchDiagRedRangeOk` / `LaunchDiagBlueRangeOk`: physics-frame shooter-target
  pairs inside the missile range window.
- `LaunchDiagRedAoOk` / `LaunchDiagBlueAoOk`: pairs satisfying the AO gate.
- `LaunchDiagRedTaOk` / `LaunchDiagBlueTaOk`: pairs satisfying the TA gate.
- `LaunchDiagRedGeometryOk` / `LaunchDiagBlueGeometryOk`: pairs satisfying
  range, AO, and TA together.
- `LaunchDiagRedLockMature` / `LaunchDiagBlueLockMature`: geometry-selected
  pairs whose lock timer has reached the launch delay.
- `LaunchDiagRedCooldownBlocked` / `LaunchDiagBlueCooldownBlocked`: mature locks
  blocked by launch cooldown.
- `LaunchDiagRedKillCooldownBlocked` / `LaunchDiagBlueKillCooldownBlocked`:
  mature locks blocked by kill cooldown.
- `LaunchDiagRedEngagedBlocked` / `LaunchDiagBlueEngagedBlocked`: alive enemy
  pairs skipped because the target is already engaged.
- `LaunchDiagRedLaunches` / `LaunchDiagBlueLaunches`: launches recorded by the
  same automatic launch path as the environment.

Derived rates:

- `RedGeometryToLaunchRate`, `BlueGeometryToLaunchRate`
- `RedRangeToGeometryRate`, `BlueRangeToGeometryRate`

Use these fields to distinguish "never entered launch geometry" from
"geometry existed but lock/cooldown/deconfliction prevented firing."

## 22. BRMA standalone loss static smoke

This command does not create the environment, reset JSBSim, train, or evaluate:

```powershell
python scripts/smoke_brma_losses_static.py
python scripts/smoke_brma_collection_soft_path.py
python scripts/smoke_brma_train_step_static.py
python scripts/smoke_brma_train_mode_static.py
```

Local BRMA train-mode smoke command, not run by Codex because it starts the
environment:

```powershell
python train_attention_mappo.py --preset attention_1v1_strict_eq33_attncritic_brma_train_smoke
```

## 23. BRMA paper reproduction presets

Smoke presets only validate that the code path starts and logs correctly. They
are not paper results.

The formal 2v2 reproduction candidates are:

```powershell
python train_attention_mappo.py --preset attention_2v2_brma_paper_main
python train_attention_mappo.py --preset attention_2v2_attn_nobrma_paper_baseline
```

The 500k probes are early health checks only:

```powershell
python train_attention_mappo.py --preset attention_2v2_brma_paper_500k_probe
python train_attention_mappo.py --preset attention_2v2_attn_nobrma_paper_500k_probe
```
