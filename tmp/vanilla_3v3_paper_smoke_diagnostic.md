# REVIEW

PASS 只代表训练链路和数值行为允许继续长训练，不代表算法已经学会空战。

## 判定原因
- REVIEW: ActionStdMean、PolicyEntropy 和越界率均为正斜率

## 指标摘要

|Metric|First|Last|Last10Mean|Min|Max|Slope|
|---|---:|---:|---:|---:|---:|---:|
|ActorLoss|-0.052356|-0.069017|-0.152752|-0.764856|0.402754|-9.22141e-06|
|PolicyLoss|-0.01995|-0.031466|-0.115413|-0.731964|0.437423|-6.40479e-06|
|EntropyBonus|0.032405|0.03755|0.0373385|0.032405|0.03755|2.81626e-06|
|PolicyEntropy|0.648107|0.751006|0.746776|0.648107|0.751006|5.63262e-05|
|CriticLoss|3877.94|927.58|946.023|447.454|4180.89|-1.02318|
|ActionStdMean|0.300877|0.311154|0.310799|0.300877|0.311154|5.7237e-06|
|ActionStdMin|0.299674|0.307854|0.307728|0.29868|0.307854|4.81825e-06|
|ActionStdMax|0.302179|0.315806|0.315535|0.302179|0.315806|7.5999e-06|
|ActionLogStdMean|-1.20106|-1.16753|-1.16867|-1.20106|-1.16753|1.86978e-05|
|ActionStdDeltaFromInit|0.000877|0.011154|0.010799|0.000877|0.011154|5.7237e-06|
|ActionStdGrowthRatio|0.300877|0.311154|0.310799|0.300877|0.311154|5.7237e-06|
|RawActionOutOfBoundsFrac|0|0.004444|0.0041015|0|0.007984|1.30738e-06|
|RawActionOutOfBoundsFracPitch|0|0.01|0.005|0|0.01|2.74159e-06|
|RawActionOutOfBoundsFracHeading|0|0|0.0018795|0|0.011976|1.46624e-07|
|RawActionOutOfBoundsFracVelocity|0|0.003333|0.005426|0|0.011976|1.03464e-06|
|EnvActionNearBoundFrac|0|0.004444|0.0041015|0|0.007984|1.30738e-06|
|EnvActionNearBoundFracPitch|0|0.01|0.005|0|0.01|2.74159e-06|
|EnvActionNearBoundFracHeading|0|0|0.0018795|0|0.011976|1.46624e-07|
|EnvActionNearBoundFracVelocity|0|0.003333|0.005426|0|0.011976|1.03464e-06|
|ActorUpdatesSkipped|0|0|0|0|0|0|
|CriticUpdatesSkipped|0|0|0|0|0|0|
|RedMeanReward|0|-391.577|-391.577|-397.615|0|-0.0560455|
|WinRateRecent|0|0|0|0|0|0|
|RedWinRate|0|0|0|0|0|0|
|Episodes|0|18|18|0|18|0.00959398|
|RedWins|0|0|0|0|0|0|
|BlueWins|0|18|18|0|18|0.00959398|
|Draws|0|0|0|0|0|0|
|RedMissiles|0|0|0|0|0|0|
|BlueMissiles|0|3.3|3.3|0|3.4|0.000611278|
|LaunchDiagRedGeometryOk|0|0|0|0|0|0|
|LaunchDiagBlueGeometryOk|80|172|86|0|172|0.00635338|
|LaunchDiagRedLockMature|0|0|0|0|0|0|
|LaunchDiagBlueLockMature|4|6|3|0|17|-0.000240602|
|LaunchDiagRedLaunches|0|0|0|0|0|0|
|LaunchDiagBlueLaunches|4|6|3|0|6|0.000172932|
|RedMissileHitRate|0|0|0|0|0|0|
|BlueMissileHitRate|0|0.830769|0.873012|0|1|0.000183311|
|RedDeathsMissile|0|54|54|0|54|0.028782|
|RedDeathsCrash|0|0|0|0|0|0|
|BlueDeathsMissile|0|0|0|0|0|0|
|BlueDeathsCrash|0|0|0|0|0|0|

## 累计字段末行值

- BlueDeathsMissile: 0
- BlueWins: 18
- Draws: 0
- Episodes: 18
- RedDeathsMissile: 54
- RedWins: 0
