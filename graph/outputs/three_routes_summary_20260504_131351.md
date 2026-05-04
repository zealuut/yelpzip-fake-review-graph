# Three Routes Summary

## Route A: old current top-k vs current EGAT

route,output_dir,edge_set,backbone,relation_model,AUC,AP,F1,Recall,Precision
A,/home/xyz/HuChao (2)/Bert-TextClassification/graph/outputs/routeA_current_topk_egat_20260503_002004/A0_current_EGAT_Base_CB,Base_CB,current_egat,edge_aware_gat,0.8518259610824274,0.8474090197070194,0.7772446881425634,0.8500749625187406,0.7159090909090909
A,/home/xyz/HuChao (2)/Bert-TextClassification/graph/outputs/routeA_current_topk_egat_20260503_002004/A1_current_EGAT_Base_LogicAE_CB,Base_LogicAE_CB,current_egat,edge_aware_gat,0.8555999361638522,0.8564474906285124,0.7707049965776865,0.8440779610194903,0.7090680100755667
A,/home/xyz/HuChao (2)/Bert-TextClassification/graph/outputs/routeA_current_topk_egat_20260503_002004/A2_current_EGAT_Full,Full,current_egat,edge_aware_gat,0.8454625760582977,0.8421998380125781,0.7698686938493434,0.8350824587706147,0.7141025641025641

## Route B: SeniorBaseExact vs SeniorBaseExact + LogicAE_CB vs SeniorBaseExact + Full

No completed outputs found.

## Route C: LogicAE as edge vs abnormal score as edge weight/gate/attention bias

No completed outputs found.
