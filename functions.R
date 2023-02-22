# apply h2o random forest
applyH2ORandomForest <- function(trainData, testData, iterVal,sIterVal){
  
  perfMat <- data.frame()
  
  # create training task
  trainTask <- makeClassifTask(data = trainData,target = "op24", 
                               positive = 1, fixup.data = "no", 
                               check.data = FALSE)
  # create test task
  testTask <- makeClassifTask(data = testData, target = "op24", 
                              positive = 1, fixup.data = "no", 
                              check.data = FALSE)
  
  # create a h2o random forest learner
  rf <- makeLearner("classif.h2o.randomForest", predict.type = "prob")

  # set range of parameter values for hyper parameter tuning
  rf_param <- makeParamSet(
    #makeIntegerParam("ntree",lower = 100, upper = 500),
    makeNumericParam("sample_rate", lower = 0.01, upper = 1),
    makeIntegerParam("ntrees", lower = 10, upper = 500),
    makeIntegerParam("max_depth", lower = 3, upper = 10),
    makeIntegerParam("nbins", lower = 2, upper = 20),
    makeIntegerParam("min_rows", lower = 5, upper = 50),
    makeIntegerParam("nbins_cats", lower = 100, upper = 2000)
  )
  
  # set strategy for optimal parameter selection
  rancontrol <- makeTuneControlRandom(maxit = 50L)
  
  # set cross validation
  set_cv <- makeResampleDesc("CV",iters = 5L)
  
  rf_tune <- tuneParams(learner = rf, resampling = set_cv, 
                        task = trainTask, par.set = rf_param, 
                        control = rancontrol, measures = acc)
  
  # store the parameter tuning results
  perfMat[iterVal,"iteration"] <- iterVal
  perfMat[iterVal,"sampling_iteration"] <- sIterVal
  perfMat[iterVal,"CV_meanAccuracy"] <- as.numeric(rf_tune$y)
  perfMat[iterVal,"sample_rate"] <- rf_tune$x$sample_rate
  perfMat[iterVal,"ntrees"] <- rf_tune$x$ntrees
  perfMat[iterVal,"max_depth"] <- rf_tune$x$max_depth
  perfMat[iterVal,"nbins"] <- rf_tune$x$nbins
  perfMat[iterVal,"min_rows"] <- rf_tune$x$min_rows
  perfMat[iterVal,"nbins_cats"] <- rf_tune$x$nbins_cats
  
  # using hyper parameters for modeling
  rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)
  rforest <- train(rf.tree, trainTask)
  rfmodelTrain <- predict(rforest, trainTask)
  rfmodelTest <- predict(rforest, testTask)
  save(rfmodelTest,file = paste0(getwd(),"/Analysis/finalV4_100iterations/h2OrandomForest/truthResponse_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  
  # evaluate the performance
  confStats <- performance(rfmodelTest, measures = list(fpr, tpr, acc, ppv))
  perfMat[iterVal,"testData_fpr"] <- as.numeric(confStats[1])
  perfMat[iterVal,"testData_tpr"] <- as.numeric(confStats[2])
  perfMat[iterVal,"testData_acc"] <- as.numeric(confStats[3])
  perfMat[iterVal,"testData_ppv"] <- as.numeric(confStats[4])
  
  d = generateThreshVsPerfData(list(train = rfmodelTrain,test = rfmodelTest), measures = list(fpr, tpr, ppv))
  plotThreshVsPerf(d)
  
  rocTrain <- paste0("Train: ",round(mlr::performance(rfmodelTrain, mlr::auc),2))
  rocTest <- paste0("Test: ",round(mlr::performance(rfmodelTest, mlr::auc),2))
  rocPlot <- plotROCCurves(d) + scale_color_discrete(breaks=c("test","train"),
                                       labels = c(rocTest, rocTrain)) + theme_bw()

  ggsave(rocPlot,filename = paste0(getwd(),"/Analysis/finalV4_100iterations/h2OrandomForest/rocPlot_iteration",iterVal,"_samplingIteration",sIterVal,".pdf"))
  aucROC <- mlr::performance(rfmodelTest, mlr::auc)
  perfMat[iterVal,"testData_aucROC"] <- as.numeric(aucROC)
  
  prcTrain <- paste0("Train: ",round(mlr3measures::prauc(rfmodelTrain$data$truth,rfmodelTrain$data$prob.1,"1"),2))
  prcTest <- paste0("Test: ",round(mlr3measures::prauc(rfmodelTest$data$truth,rfmodelTest$data$prob.1,"1"),2))
  prcPlot <- plotROCCurves(d, measures = list(tpr, ppv), diagonal = FALSE) + 
    scale_color_discrete(breaks=c("test","train"), labels = c(prcTest, prcTrain)) + theme_bw()
  
  ggsave(prcPlot,filename = paste0(getwd(),"/Analysis/finalV4_100iterations/h2OrandomForest/prcPlot_iteration",iterVal,"_samplingIteration",sIterVal,".pdf"))
  aucPRC <- mlr3measures::prauc(rfmodelTest$data$truth,rfmodelTest$data$prob.1,"1")
  perfMat[iterVal,"testData_aucPRC"] <- as.numeric(aucPRC)

  r = calculateROCMeasures(rfmodelTest)
  cMat <- calculateConfusionMatrix(rfmodelTest)
  save(cMat,file = paste0(getwd(),"/Analysis/finalV4_100iterations/h2OrandomForest/confusionMatrix_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  feat <- getFeatureImportance(rforest)
  save(feat,file = paste0(getwd(),"/Analysis/finalV4_100iterations/h2OrandomForest/featureImportance_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  
  finalRes <- list(train = rfmodelTrain, test = rfmodelTest,
                   perfMat = perfMat)
  return(finalRes)
  
}

# apply cforest 
applyCForest <- function(trainData, testData, iterVal, sIterVal){
  
  perfMat <- data.frame()
  
  # create training task
  trainTask <- makeClassifTask(data = trainData,target = "op24", 
                               positive = 1, fixup.data = "no", 
                               check.data = FALSE)
  # create test task
  testTask <- makeClassifTask(data = testData, target = "op24", 
                              positive = 1, fixup.data = "no", 
                              check.data = FALSE)
  # create a cforest learner
  rf <- makeLearner("classif.cforest", predict.type = "prob", 
                    par.vals = list(ntree = 500, mtry = 3))
  
  # set range of parameter values for hyper parameter tuning
  rf_param <- makeParamSet(
    makeIntegerParam("ntree",lower = 100, upper = 500),
    makeIntegerParam("mtry", lower = 3, upper = 10)
    
  )
  
  # set strategy for optimal parameter selection
  rancontrol <- makeTuneControlRandom(maxit = 50L)
  # set cross validation
  set_cv <- makeResampleDesc("CV",iters = 5L)
  
  # hyper parameter tuning and cross validation
  rf_tune <- tuneParams(learner = rf, resampling = set_cv, 
                        task = trainTask, par.set = rf_param, 
                        control = rancontrol, measures = acc)
  
  # store the parameter tuning results
  perfMat[iterVal,"iteration"] <- iterVal
  perfMat[iterVal,"sampling_iteration"] <- sIterVal
  perfMat[iterVal,"CV_meanAccuracy"] <- as.numeric(rf_tune$y)
  perfMat[iterVal,"ntree"] <- rf_tune$x$ntree
  perfMat[iterVal,"mtry"] <- rf_tune$x$mtry
  
  # using hyper parameters for modeling
  rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)
  rforest <- train(rf.tree, trainTask)
  rfmodelTrain <- predict(rforest, trainTask)
  rfmodelTest <- predict(rforest, testTask)
  save(rfmodelTest,file = paste0(getwd(),"/Analysis/finalV4_100iterations/cforest/truthResponse_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  
  # evaluate the performance
  confStats <- performance(rfmodelTest, measures = list(fpr, tpr, acc, ppv))
  perfMat[iterVal,"testData_fpr"] <- as.numeric(confStats[1])
  perfMat[iterVal,"testData_tpr"] <- as.numeric(confStats[2])
  perfMat[iterVal,"testData_acc"] <- as.numeric(confStats[3])
  perfMat[iterVal,"testData_ppv"] <- as.numeric(confStats[4])
  
  d = generateThreshVsPerfData(list(train = rfmodelTrain,test = rfmodelTest), measures = list(fpr, tpr, ppv))
  plotThreshVsPerf(d)
  
  rocTrain <- paste0("Train: ",round(mlr::performance(rfmodelTrain, mlr::auc),2))
  rocTest <- paste0("Test: ",round(mlr::performance(rfmodelTest, mlr::auc),2))
  rocPlot <- plotROCCurves(d) + scale_color_discrete(breaks=c("test","train"),
                                                     labels = c(rocTest, rocTrain)) + theme_bw()

  ggsave(rocPlot,filename = paste0(getwd(),"/Analysis/finalV4_100iterations/cforest/rocPlot_iteration",iterVal,"_samplingIteration",sIterVal,".pdf"))
  aucROC <- mlr::performance(rfmodelTest, mlr::auc)
  perfMat[iterVal,"testData_aucROC"] <- as.numeric(aucROC)
  
  prcTrain <- paste0("Train: ",round(mlr3measures::prauc(rfmodelTrain$data$truth,rfmodelTrain$data$prob.1,"1"),2))
  prcTest <- paste0("Test: ",round(mlr3measures::prauc(rfmodelTest$data$truth,rfmodelTest$data$prob.1,"1"),2))
  prcPlot <- plotROCCurves(d, measures = list(tpr, ppv), diagonal = FALSE) + 
    scale_color_discrete(breaks=c("test","train"), labels = c(prcTest, prcTrain)) + theme_bw()
  
  ggsave(prcPlot,filename = paste0(getwd(),"/Analysis/finalV4_100iterations/cforest/prcPlot_iteration",iterVal,"_samplingIteration",sIterVal,".pdf"))
  aucPRC <- mlr3measures::prauc(rfmodelTest$data$truth,rfmodelTest$data$prob.1,"1")
  perfMat[iterVal,"testData_aucPRC"] <- as.numeric(aucPRC)
  
  r = calculateROCMeasures(rfmodelTest)
  cMat <- calculateConfusionMatrix(rfmodelTest)
  save(cMat,file = paste0(getwd(),"/Analysis/finalV4_100iterations/cforest/confusionMatrix_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  feat <- getFeatureImportance(rforest)
  save(feat,file = paste0(getwd(),"/Analysis/finalV4_100iterations/cforest/featureImportance_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  
  finalRes <- list(train = rfmodelTrain, test = rfmodelTest,
                   perfMat = perfMat)
  return(finalRes)
  
}

# apply gradient boosting
applyXgBoost <- function(trainData, testData, iterVal, sIterVal){
  
  perfMat <- data.frame()
  
  # create training task
  trainTask <- makeClassifTask(data = trainData,target = "op24", 
                               positive = 1, fixup.data = "no", 
                               check.data = FALSE)
  # create test task
  testTask <- makeClassifTask(data = testData, target = "op24", 
                              positive = 1, fixup.data = "no", 
                              check.data = FALSE)
  
  trainTask <- createDummyFeatures(trainTask)
  testTask <- createDummyFeatures(testTask)
  
  # create xgboost learner
  xgb_learner <- makeLearner(
    "classif.xgboost",
    predict.type = "prob",
    par.vals = list(
      objective = "binary:logistic",
      eval_metric = "error",
      nrounds = 200
    )
  )
  
  
  # set range of parameter values for hyper parameter tuning
  xgb_params <- makeParamSet(
    # The number of trees in the model (each one built sequentially)
    makeIntegerParam("nrounds", lower = 100, upper = 500),
    # number of splits in each tree
    makeIntegerParam("max_depth", lower = 1, upper = 10),
    # "shrinkage" - prevents overfitting
    makeNumericParam("eta", lower = .1, upper = .5),
    # L2 regularization - prevents overfitting
    makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
  )
  
  # set strategy for optimal parameter selection
  rancontrol <- makeTuneControlRandom(maxit = 50L)
  
  # set cross validation
  set_cv <- makeResampleDesc("CV",iters = 5L)
  
  # hyper parameter tuning and cross validation
  xgB_tune <- tuneParams(learner = xgb_learner, resampling = set_cv, 
                         task = trainTask, par.set = xgb_params, 
                         control = rancontrol, measures = acc)
  
  # store the parameter tuning results
  perfMat[iterVal,"iteration"] <- iterVal
  perfMat[iterVal,"sampling_iteration"] <- sIterVal
  perfMat[iterVal,"CV_meanAccuracy"] <- as.numeric(xgB_tune$y)
  perfMat[iterVal,"nrounds"] <- xgB_tune$x$nrounds
  perfMat[iterVal,"max_depth"] <- xgB_tune$x$max_depth
  perfMat[iterVal,"eta"] <- xgB_tune$x$eta
  perfMat[iterVal,"lambda"] <- xgB_tune$x$lambda
  
  # using hyper parameters for modeling
  xgb.tree <- setHyperPars(xgb_learner, par.vals = xgB_tune$x)
  xgBst <- train(xgb.tree, trainTask)
  rfmodelTrain <- predict(xgBst, trainTask)
  rfmodelTest <- predict(xgBst, testTask)
  save(rfmodelTest,file = paste0(getwd(),"/Analysis/finalV4_100iterations/xgboost/truthResponse_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  
  # evaluate the performance
  confStats <- performance(rfmodelTest, measures = list(fpr, tpr, acc, ppv))
  perfMat[iterVal,"testData_fpr"] <- as.numeric(confStats[1])
  perfMat[iterVal,"testData_tpr"] <- as.numeric(confStats[2])
  perfMat[iterVal,"testData_acc"] <- as.numeric(confStats[3])
  perfMat[iterVal,"testData_ppv"] <- as.numeric(confStats[4])
  
  d = generateThreshVsPerfData(list(train = rfmodelTrain,test = rfmodelTest), 
                               measures = list(fpr, tpr, ppv))
  plotThreshVsPerf(d)
  
  rocTrain <- paste0("Train: ",round(mlr::performance(rfmodelTrain, mlr::auc),2))
  rocTest <- paste0("Test: ",round(mlr::performance(rfmodelTest, mlr::auc),2))
  rocPlot <- plotROCCurves(d) + scale_color_discrete(breaks=c("test","train"),
                                                     labels = c(rocTest, rocTrain)) + theme_bw()
  #ggsave(rocPlot,filename = "test.pdf")
  ggsave(rocPlot,filename = paste0(getwd(),"/Analysis/finalV4_100iterations/xgboost/rocPlot_iteration",iterVal,"_samplingIteration",sIterVal,".pdf"))
  aucROC <- mlr::performance(rfmodelTest, mlr::auc)
  perfMat[iterVal,"testData_aucROC"] <- as.numeric(aucROC)
  
  prcTrain <- paste0("Train: ",round(mlr3measures::prauc(rfmodelTrain$data$truth,rfmodelTrain$data$prob.1,"1"),2))
  prcTest <- paste0("Test: ",round(mlr3measures::prauc(rfmodelTest$data$truth,rfmodelTest$data$prob.1,"1"),2))
  prcPlot <- plotROCCurves(d, measures = list(tpr, ppv), diagonal = FALSE) + 
    scale_color_discrete(breaks=c("test","train"), labels = c(prcTest, prcTrain)) + theme_bw()
  
  ggsave(prcPlot,filename = paste0(getwd(),"/Analysis/finalV4_100iterations/xgboost/prcPlot_iteration",iterVal,"_samplingIteration",sIterVal,".pdf"))
  aucPRC <- mlr3measures::prauc(rfmodelTest$data$truth,rfmodelTest$data$prob.1,"1")
  perfMat[iterVal,"testData_aucPRC"] <- as.numeric(aucPRC)
  
  r = calculateROCMeasures(rfmodelTest)
  cMat <- calculateConfusionMatrix(rfmodelTest)
  save(cMat,file = paste0(getwd(),"/Analysis/finalV4_100iterations/xgboost/confusionMatrix_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  feat <- getFeatureImportance(xgBst)
  save(feat,file = paste0(getwd(),"/Analysis/finalV4_100iterations/xgboost/featureImportance_iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
  
  finalRes <- list(train = rfmodelTrain, test = rfmodelTest,
                   perfMat = perfMat)
  return(finalRes)
  
}


