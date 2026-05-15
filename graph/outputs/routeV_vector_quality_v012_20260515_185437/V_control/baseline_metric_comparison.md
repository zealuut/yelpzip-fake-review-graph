# Route Baseline Fresh Retrain Best-of-5

Primary route baseline: best completed full yelpcurrent artifact retrain plus RouteD/D1 graph-stage rerun.

| name | promoted | review_epoch | review_val_auc | d1_auc | d1_ap | d1_f1 | d1_recall | d1_precision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repeat01 | yes | 1 | 0.7496077001 | 0.8545569793813736 | 0.8522126584244006 | 0.7827278958190541 | 0.856071964017991 | 0.7209595959595959 | Complete fresh artifact retrain followed by RouteD/D1 graph-stage rerun. |
| repeat02 | no | 2 | 0.7550083385 | 0.846498789585717 | 0.8406636596031787 | 0.7805519053876478 | 0.8905547226386806 | 0.6947368421052632 | Complete fresh artifact retrain followed by RouteD/D1 graph-stage rerun. |
| repeat03 | no | 3 | 0.7542490813 | 0.7954658352982429 | 0.7778665430759033 | 0.7323943661971831 | 0.7796101949025487 | 0.6905710491367862 | Complete fresh artifact retrain followed by RouteD/D1 graph-stage rerun. |
| repeat04 | no | 2 | 0.7519142377 | 0.8469932949567195 | 0.8423403896724251 | 0.7773333333333333 | 0.8740629685157422 | 0.6998799519807923 | Complete fresh artifact retrain followed by RouteD/D1 graph-stage rerun. |
| repeat05 | no | 3 | 0.755290071 | 0.7978349655756829 | 0.7830245405249765 | 0.7346101231190151 | 0.8050974512743628 | 0.6754716981132075 | Complete fresh artifact retrain followed by RouteD/D1 graph-stage rerun. |
| D1_STRONGEST_GRAPH_ONLY_0P8563709149922789 | no | None | None | 0.8563709149922789 | 0.858368711617606 | 0.7781715095676824 | None | None | Audit/reference only; not the primary route baseline because it is graph-only from a fixed artifact. |

Notes:
- The fixed-artifact historical strongest baseline is retained as an audit reference only.
- The promoted row comes from a full retraining chain, then a D1 graph-stage rerun.
