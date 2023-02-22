library(mlr)
library(ggplot2)
library(party)
library(h2o)
library(openxlsx)
library(dplyr)

source("/mnt/disk1/2022_09_PainPrediction_Anshupa_Svetozar_Fahrrad_Andreas/scripts/scripts/functionsV2.R")
#source("C:/CUBA_Projects/2022_09_PainPrediction_Anshupa_Svetozar_Fahrrad_Andreas/functions.R")

set.seed(12)
#h2o.init(port = 44444)
painData <- openxlsx::read.xlsx("/mnt/disk1/2022_09_PainPrediction_Anshupa_Svetozar_Fahrrad_Andreas/Documents/pain2.xlsx",
                                na.strings = c("NA","Na","A","AN","   "))
painData$wbc <- as.numeric(painData$wbc)
painData$lactate <- as.numeric(painData$lactate)

painData[which(painData$fever %in% c("42","2")),"fever"] <- "0"
painData[which(painData$pain_rlr == "ß"),"pain_rlr"] <- "0"

# get all columns with continuous/numeric value
numCols <- colnames(painData %>% select_if(function(col) length(unique(col)) > 3))

# get column with discrete values (0 or 1)
disCols <- colnames(painData %>% select_if(function(col) length(unique(col)) <= 3))
# convert these columns to factor
painData[disCols] <- lapply(painData[disCols], factor)

perfMatH2O <- data.frame()
perfCforest <- data.frame()
perfXgBoost <- data.frame()

for(iterVal in 17:20){
  sampledIdx <- sample(1:nrow(painData),replace = FALSE)
  samplingList <- split(sampledIdx, ceiling(seq_along(sampledIdx)/(round(nrow(painData)/5)+1)))
  testListVal <- sample(1:length(samplingList),5)
  
  for(sIterVal in 1:length(testListVal)){
    testListIdx <- testListVal[sIterVal]
    # split data 80-20 split
    testData <- painData[samplingList[[testListIdx]],]
    save(testData,file = paste0(getwd(),"/Analysis/finalV4_100iterations/datasets/testData_Iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
    
    trainListIdx <- setdiff(1:length(samplingList),testListIdx)
    trainData <- painData[unlist(samplingList[trainListIdx]),]
    save(trainData,file = paste0(getwd(),"/Analysis/finalV4_100iterations/datasets/trainData_Iteration",iterVal,"_samplingIteration",sIterVal,".RData"))
    
    # run default random forest using h2o package
    resH2O <- applyH2ORandomForest(trainData = trainData, testData = testData, iterVal = iterVal,sIterVal = sIterVal)
    perfMatH2O <- rbind(perfMatH2O,resH2O$perfMat)
    perfMatH2O <- perfMatH2O[!is.na(perfMatH2O$iteration),]
    rownames(perfMatH2O) <- 1:nrow(perfMatH2O)
    
    # run cforest
    resCForest <- applyCForest(trainData = trainData, testData = testData, iterVal = iterVal,sIterVal = sIterVal)
    perfCforest <- rbind(perfCforest,resCForest$perfMat)
    perfCforest <- perfCforest[!is.na(perfCforest$iteration),]
    rownames(perfCforest) <- 1:nrow(perfCforest)
    
    # run xgboost
    resXgBoost <- applyXgBoost(trainData = trainData, testData = testData, iterVal = iterVal,sIterVal = sIterVal)
    perfXgBoost <- rbind(perfXgBoost,resXgBoost$perfMat)
    perfXgBoost <- perfXgBoost[!is.na(perfXgBoost$iteration),]
    rownames(perfXgBoost) <- 1:nrow(perfXgBoost)
    
    df = generateThreshVsPerfData(list(h2ORandom = resH2O$test, 
                                       cforest = resCForest$test,
                                       xgBoost = resXgBoost$test),
                                  measures = list(fpr, tpr, ppv))
    rocCForest <- paste0("cForest: ",round(mlr::performance(resCForest$test, mlr::auc),3))
    rocH20Random <- paste0("h2ORandom: ",round(mlr::performance(resH2O$test, mlr::auc),3))
    rocXgBoost <- paste0("xgBoost: ",round(mlr::performance(resXgBoost$test, mlr::auc),3))
    
    fPlot <- plotROCCurves(df) + 
      scale_color_discrete(breaks=c("cforest","h2ORandom","xgBoost"),
                           labels = c(rocCForest, rocH20Random, rocXgBoost)) + 
      theme_bw()
    
    ggsave(fPlot,filename = paste0(getwd(),"/Analysis/finalV4_100iterations/rocPlot_iteration",iterVal,"_samplingIteration",sIterVal,".pdf"),
           units = "cm",width = 13,height = 7)
    
    prcCForest <- paste0("cForest: ",round(mlr3measures::prauc(resCForest$test$data$truth,resCForest$test$data$prob.1,"1"),3))
    prcH20Random <- paste0("h2ORandom: ",round(mlr3measures::prauc(resH2O$test$data$truth,resH2O$test$data$prob.1,"1"),3))
    prcXgBoost <- paste0("xgBoost: ",round(mlr3measures::prauc(resXgBoost$test$data$truth,resXgBoost$test$data$prob.1,"1"),3))
    
    yLineValue <- nrow(testData[which(testData$op24 == 1),])/nrow(testData)
    fPRCplot <- plotROCCurves(df, measures = list(tpr, ppv), diagonal = FALSE) + 
      geom_hline(yintercept = yLineValue, linetype='dashed')+
      scale_color_discrete(breaks = c("cforest","h2ORandom","xgBoost"),
                           labels = c(prcCForest, prcH20Random, prcXgBoost)) + 
      theme_bw()
    
    ggsave(fPRCplot,filename = paste0(getwd(),"/Analysis/finalV4_100iterations/prcPlot_iteration",iterVal,"_samplingIteration",sIterVal,".pdf"),
           units = "cm",width = 13,height = 7)
  }
}

perfList <- list("h2ORandom" = perfMatH2O, "cForest" = perfCforest, "xgBoost" = perfXgBoost)
openxlsx::write.xlsx(perfList, file = paste0(getwd(),"/Analysis/finalV4_100iterations/performanceReport.xlsx"))

