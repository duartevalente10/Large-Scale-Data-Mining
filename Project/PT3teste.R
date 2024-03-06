library(dplyr) 
library(sparklyr) 
library(data.table) 
library(smotefamily)
library(tidyr)
library(class)

# funções auxiliares
if(!exists("printConfusionMatrix", mode="function")) 
  source("helperfunctions.R")

# spark setup 
spark_disconnect_all() #just preventive code
sc <- spark_connect('local', version = '2.3.0', hadoop_version = '2.7', config = list())

# carregar dados
basepath <- "Data/Influenza-Outbreak-Dataset"
tr.data <- c("train_data_1.csv", "train_data_2.csv", "train_data_3.csv", "train_data_4.csv", "train_data_5.csv") # data
labels<- c("train_labels_1.csv", "train_labels_2.csv", "train_labels_3.csv", "train_labels_4.csv", "train_labels_5.csv") # labels

# funções para ler e formatar os dados 
fun1 <- function(i) { 
  read.csv(paste(basepath,"train",i,sep = "/"), header=FALSE,stringsAsFactors = FALSE)
}

fun2 <- function(i) { 
  read.csv(paste(basepath,"train",i,sep = "/"), header=FALSE,stringsAsFactors = FALSE) %>% t %>% as.data.table
}

df<-do.call(rbind, lapply(tr.data, fun1 )) #bind csv together
df.l<-do.call(rbind, lapply(labels, fun2 )) #bind class together
names(df.l) <-c("CLASS") #rename dependent variable
df.local<- cbind(df.l,df) #bind them together

# copia os dados para spark dataframe
df <- copy_to(sc, df.local)

#----------------Feature Selection---------------------------

# selecionadas as features mais relevantes calculadas na fase anterior
idx <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,28,30,37,39,41,42,49,51,60,62,70,81,83,90,91,136,137,148,164,175,286,301,385)
# selecionar as colunas com os ids selecionados
df.sel <- df.local %>% select(all_of(idx))

# seed para calculos 
set.seed(123)

#passar para dataframe em R
dataf<- as.data.frame(df.sel)

#passar para dataframe no Spark
my_spark_data <- copy_to(sc, dataf, "my_table_name1")

#----------------Split---------------------------

#dividir os dados em treino e testes 
splits <- my_spark_data %>% sdf_random_split(training = 0.80, testing = 0.20)
sdf_train <- splits$training
sdf_test <- splits$testing

# voltar a transformar em Rdataframe
sdf_train_data <- collect(sdf_train)
sdf_test_data <- collect(sdf_test)

# classes de cada dataset
table(sdf_train_data$CLASS) # 4224 da classe 0 e 170 da classe 1
table(sdf_test_data$CLASS) # 1038 da classe 0 e 43 da classe 1

#----------------Treino sem balanciamento-------------------

# modelo sem balanciamento de dados
rf_model <- ml_random_forest(sdf_train, formula = "CLASS ~ .") # pretendemos prever a classe da instancia com nas restantes features 
predictions <- ml_predict(rf_model, sdf_test) # prever os resultados
mdle.printConfusionMatrix(predictions) # Accuary 0.959 / False Positive Rate : 0.933


# -------- Mean Square Error --------------------
# Extract the predicted and actual values
predicted_values <- select(predictions, "prediction") %>% collect() %>% pull()
actual_values <- select(sdf_test, "CLASS") %>% collect() %>% pull()

# Calculate the squared differences
squared_diff <- (predicted_values - actual_values) ^ 2

# Compute the mean squared error
mse <- mean(squared_diff)
mse

# ------------- Correlation---------------------
# Calculate the correlation coefficient
correlation <- cor(predicted_values, actual_values)

correlation

# ----------- Root Mean Square Error ----------

# Calculate the squared differences
squared_diff <- (predicted_values - actual_values) ^ 2

# Compute the mean squared error
mse <- mean(squared_diff)

# Calculate the root mean squared error
rmse <- sqrt(mse)

rmse

# ------------ Mean ablsolute error------------

# Calculate the absolute differences
absolute_diff <- abs(predicted_values - actual_values)

# Compute the mean absolute error
mae <- mean(absolute_diff)

mae

# ------------ Mean absolute relative error -------------
# Calculate the absolute differences
absolute_diff <- abs(predicted_values - actual_values)

# Compute the mean absolute error (MAE)
mae <- mean(absolute_diff)

# Compute the mean absolute actual value (MAAV)
maav <- mean(abs(actual_values))

# Compute the relative absolute error (RAE)
rae <- mae / maav

rae

# -----precision-------------

pred<-predictions %>% select('CLASS', 'prediction')  %>% collect
pred$prediction<-round(pred$prediction)
cfxmat<-confusionMatrix(table(as.vector(pred$CLASS),as.vector(pred$prediction)),mode = "everything",positive = "1")

# Extract the true positives (TP), false positives (FP), and false negatives (FN)
tp <- cfxmat$table[2, 2]  # True Positives
fp <- cfxmat$table[1, 2]  # False Positives
fn <- cfxmat$table[2, 1]  # False Negatives

# Compute the precision
precision <- tp / (tp + fp)

precision
# ----- recall -------------

# Compute the recall
recall <- tp / (tp + fn)

recall

# -------Model avaliation ---------------
print('Mean Square error: ')
mse
print('Root Mean Square Error: ')
rmse
print('Mean ablsolute error: ')
mae
print('Mean absolute relative error: ')
rae
print('Correlation: ')
correlation

# -----------------------------------------------
#----------------Undersampling-------------------
# -----------------------------------------------

# efetuar undersampling nos dados
df_pos <- sdf_bind_rows(filter(sdf_train, CLASS == "1"))
df_neg <- sdf_bind_rows(filter(sdf_train, CLASS == "0"))

n_pos <- sdf_nrow(filter(sdf_train, CLASS == "1"))
n_neg <- sdf_nrow(filter(sdf_train, CLASS == "0"))

fraction <- as.numeric(n_pos) / as.numeric(n_neg)

df_balanced <- sdf_sample(df_neg, fraction)
df_balanced <- sdf_bind_rows(df_balanced, df_pos)

df_balanced_data <- collect(df_balanced) # foram reduzidas de 5375 para 342 instancias

#----------------Treino com undersampling-----------

# modelo com balanciamento de dados(undersampling)
rf_model_balanced <- ml_random_forest(df_balanced, formula = "CLASS ~ .") # pretendemos prever a classe da instancia com nas restantes features 
predictions_balanced <- ml_predict(rf_model_balanced, sdf_test) # prever os resultados
mdle.printConfusionMatrix(predictions_balanced) # Accuracy: 0.705  / False Positive Rate : 0.441


# -------- Mean Square Error --------------------
# Extract the predicted and actual values
predicted_values <- select(predictions_balanced, "prediction") %>% collect() %>% pull()
actual_values <- select(sdf_test, "CLASS") %>% collect() %>% pull()

# Calculate the squared differences
squared_diff <- (predicted_values - actual_values) ^ 2

# Compute the mean squared error
mse <- mean(squared_diff)
mse

# ------------- Correlation---------------------
# Calculate the correlation coefficient
correlation <- cor(predicted_values, actual_values)

correlation

# ----------- Root Mean Square Error ----------

# Calculate the squared differences
squared_diff <- (predicted_values - actual_values) ^ 2

# Compute the mean squared error
mse <- mean(squared_diff)

# Calculate the root mean squared error
rmse <- sqrt(mse)

rmse

# ------------ Mean ablsolute error------------

# Calculate the absolute differences
absolute_diff <- abs(predicted_values - actual_values)

# Compute the mean absolute error
mae <- mean(absolute_diff)

mae

# ------------ Mean absolute relative error -------------
# Calculate the absolute differences
absolute_diff <- abs(predicted_values - actual_values)

# Compute the mean absolute error (MAE)
mae <- mean(absolute_diff)

# Compute the mean absolute actual value (MAAV)
maav <- mean(abs(actual_values))

# Compute the relative absolute error (RAE)
rae <- mae / maav

rae

# -------Model avaliation ---------------
print('Mean Square error: ')
mse
print('Root Mean Square Error: ')
rmse
print('Mean ablsolute error: ')
mae
print('Mean absolute relative error: ')
rae
print('Correlation: ')
correlation

# -----------------------------------------------
#----------------Oversampling-------------------
# -----------------------------------------------

# numero de instancias de cada classe
class_counts <- as.data.frame(table(sdf_train_data$CLASS))

# calculo do numero pretendido de instancias
target_count <- max(class_counts$Freq)

# applicar o oversampling para cada classe
df.train_oversampled <- sdf_train_data %>% 
  group_by(CLASS) %>% 
  do(filter(., CLASS == .$CLASS %>% head(target_count))) %>% 
  group_by(CLASS) %>% 
  do(data.frame(.[rep(1:nrow(.), each = ceiling(target_count/nrow(.))), ])) %>% 
  ungroup()

# 8474 instancias

#----------------Treino com Oversampling-------------

df.train_oversampled_data <- copy_to(sc, df.train_oversampled)

rf_model_balanced2 <- ml_random_forest(df.train_oversampled_data, formula = "CLASS ~ .") # pretendemos prever a classe da instancia com nas restantes features 
predictions_balanced2 <- ml_predict(rf_model_balanced2, sdf_test) # prever os resultados
mdle.printConfusionMatrix(predictions_balanced2) # Accuracy: 0.731   / False Positive Rate : 0.441


# -------- Mean Square Error --------------------
# Extract the predicted and actual values
predicted_values <- select(predictions_balanced2, "prediction") %>% collect() %>% pull()
actual_values <- select(sdf_test, "CLASS") %>% collect() %>% pull()

# Calculate the squared differences
squared_diff <- (predicted_values - actual_values) ^ 2

# Compute the mean squared error
mse <- mean(squared_diff)
mse

# ------------- Correlation---------------------
# Calculate the correlation coefficient
correlation <- cor(predicted_values, actual_values)

correlation

# ----------- Root Mean Square Error ----------

# Calculate the squared differences
squared_diff <- (predicted_values - actual_values) ^ 2

# Compute the mean squared error
mse <- mean(squared_diff)

# Calculate the root mean squared error
rmse <- sqrt(mse)

rmse

# ------------ Mean ablsolute error------------

# Calculate the absolute differences
absolute_diff <- abs(predicted_values - actual_values)

# Compute the mean absolute error
mae <- mean(absolute_diff)

mae

# ------------ Mean absolute relative error -------------
# Calculate the absolute differences
absolute_diff <- abs(predicted_values - actual_values)

# Compute the mean absolute error (MAE)
mae <- mean(absolute_diff)

# Compute the mean absolute actual value (MAAV)
maav <- mean(abs(actual_values))

# Compute the relative absolute error (RAE)
rae <- mae / maav

rae

# -------Model avaliation ---------------
print('Mean Square error: ')
mse
print('Root Mean Square Error: ')
rmse
print('Mean ablsolute error: ')
mae
print('Mean absolute relative error: ')
rae
print('Correlation: ')
correlation
