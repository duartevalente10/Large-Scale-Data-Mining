##########################################################
#
# Mestrado em Engenharia Informática e de Computadores
# MDLE
# 
# Material de apoio
#
#########################################################

require(caret)
require(e1071)

mdle.printConfusionMatrix<-function(sp.pred)
{
  
  pred<-sp.pred %>% select('CLASS', 'prediction')  %>% collect
  pred$prediction<-round(pred$prediction)
  cfxmat<-confusionMatrix(table(as.vector(pred$CLASS),as.vector(pred$prediction)),mode = "everything",positive = "1")
  print(cfxmat$table) 
  
  cat(noquote(paste("False Positive Rate :",format(round(1-cfxmat$byClass[[3]], 3), nsmall = 3),"\n" )))
  cat(noquote(paste("Accuracy            :",format(round(cfxmat$overall[[1]], 3), nsmall = 3),"\n" )))
  cat(noquote(paste("Kappa               :",format(round(cfxmat$overall[[2]], 3), nsmall = 3),"\n" )))
  cat(noquote(paste("Pos Pred Value      :",format(round(cfxmat$byClass[[3]], 3), nsmall = 3),"\n" )))
  cat(noquote(paste("Neg Pred Value      :",format(round(cfxmat$byClass[[4]], 3), nsmall = 3) )))
  
  
}

mdle.predict<-function(model,test)
{
  return (ml_predict(model,test))
}
