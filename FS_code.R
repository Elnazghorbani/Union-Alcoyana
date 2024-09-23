library(remotes)
#install_github("David-hervas/isaves")
library(isaves)
library(clickR)
library(isaves)
load_incremental()
save_incremental()
#carga de datos
bd_segm <-read.csv("imputed_unique_clients.csv", sep=",")
descriptive(bd_segm)
#bd_segm <- bd_segm[,-95]

#Filtro para tener menos variables
#bd_segm_f <- bd_segm[,c(2:47, 92:108)]
#bd_segm_f<-fix_factors(bd_segm_f , k=11)
names (bd_segm)
#Quitamos la variable subramo por tener 1 nivel = turismo.
bd_segm$id13 <- ordered(bd_segm$id13)
summary(bd_segm)

##################################################################
#LASSO WITH ORDINAL DEPENDENT VARIABLE
##################################################################
rm(X)
bd_segm <- bd_segm[1:1000, ] #to decrease memory accupation
#print(bd_segm[,-129])


X <- model.matrix( ~ ., data=bd_segm[,-129])
colnames(bd_segm[,-129])
X <- X[,-1]
colnames(X)
summary(X)
names(bd_segm)
descriptive(bd_segm)
library(ordinalNet)
table(bd_segm$id02)
#sum(is.na(X))
#sum(is.na(bd_segm$id13))
#dim(X)
#length(bd_segm$id13)
#head(X)
#head(bd_segm$id13)
# Garbage collection to free up memory
#gc()
#X <- X[1:1000, 1:1000]  # Adjust size as needed
id13<- bd_segm$id13[1:1000]
#Ajuste del modelo
cv <- ordinalNetCV(X, id13, tuneMethod="aic")
summary(cv)

fit1 <- ordinalNet(X, bd_segm$id13, family="cumulative", link="logit",
                   parallelTerms=TRUE, nonparallelTerms=FALSE, printIter=TRUE)

#print(fit1$coefs[20,-(1:10)] )
seleccion_lasso <- fit1$coefs[20,-(1:10)] #_____


#Variables seleccionadas lasso
variables_seleccionadas_lasso <- sapply(names(bd_segm), function(x){
  any(grepl(x, names(seleccion_lasso[seleccion_lasso != 0])))
})
colnames(X)[colnames(X)%in%names(bd_segm) ]

#names(variables_seleccionadas_lasso)[variables_seleccionadas_lasso]
#sapply(colnames(X)[colnames(X)%in%names(bd_segm_f)],function(x){
#  class(bd_segm_f[,x])
#})

write.csv2(names(variables_seleccionadas_lasso)[variables_seleccionadas_lasso], file="lasso.csv")

## ##################################################################
## Perform random forest lasso
##################################################################

# install.packages("ordinalForest")
library(ordinalForest)

if ("ordinalForest" %in% rownames(installed.packages())) {
  library(ordinalForest)
} else {
  print("Package 'ordinalForest' is not installed.")
}

lasso_bd_segm_f<-bd_segm[,c("id01",
                         "id21",
                         "id24",
                         "id27",
                         "id36",
                         "id38",
                         "id57",
                         "id58_1",
                         "id61",
                         "DGC_001",
                         "DGC_002",
                         "DGC_009",
                         "DGC_011",
                         "DGC_014",
                         "DGC_016",
                         "DGC_017",
                         "DGC_018",
                         "DGT_001",
                         "DGT_002",
                         "INE_008",
                         "INE_009",
                         "INE_027",
                         "INE_034",
                         "INE_037",
                         "INE_038",
                         "INE_042",
                         "INE_051",
                         "INE_056",
                         "OSM_001",
                         "REP_005",
                         "REP_006",
                         "REP_010",
                         "id09_y",
                         "id11",
                         "id14",
                         "id17",
                         "Age",
                         "id29",
                         "id50",
                         "id52",
                         "how_to_pay",
                         "when_to_pay",
                         "id72_electric",
                         "AceptoCulpa_yes_contact",
                         "AceptoCulpa_no_contact",
                         "ContrataLunas_no_contract",
                         "ContrataLunas_yes_contract",
                         "ScoreAda",
                         "risk_0",
                         "ExposicionTotalPoliza",
                         "id13")]
#install.packages("caTools")
#require(caTools)
library(caTools)
library(vcd)
library(grid)
#class(lasso_bd_segm_f$id13)

sample = sample.split(lasso_bd_segm_f$id13, SplitRatio = .70)
train = subset(lasso_bd_segm_f, sample == TRUE)
test  = subset(lasso_bd_segm_f, sample == FALSE)

ordforlasso_70 <- ordfor(depvar="id13", data=train, nsets=50, nbest=5, ntreeperdiv=100,
                         ntreefinal=1000)
summary(ordforlasso_70)

preds70 <- predict(ordforlasso_70, newdata=test)
table(data.frame(true_values=test$id13, predictions=preds70$ypred))
agreementplot(table(data.frame(true_values=test$id13, predictions=preds70$ypred)))
estad_bandi<-agreementplot(table(data.frame(true_values=test$id13, predictions=preds70$ypred)))
estad_bandi$Bangdiwala_Weighted
#0.5641553
save_incremental()

citation("vcd")


##################################################################
## Perform stepwise selection
##################################################################
install.packages(car)
library(rms)
#bd<-bd_segm[,c(2,4:65)]

bd_segm_dense <- as.matrix(bd_segm)  # Convert sparse matrix to a dense matrix
# Identify which columns are categorical (i.e., factors or characters)
categorical_vars <- sapply(bd_segm_dense, function(x) is.factor(x) || is.character(x))

# Extract only categorical variables
bd_segm_categorical <- bd_segm_dense[, categorical_vars]


orm_model <- orm(id13 ~ ., data = bd_segm_dense)


kk <- fastbw(orm(id13 ~ ., data=bd_segm[,c(1:60)]))


kk <- fastbw(orm(id13 ~ ., data=bd_segm))


bd_segm
#kk3 <- fastbw(orm(id13 ~ ., data=bd_segm))
#[,c(1:53, 55:59, 61)]

#Variables seleccionadas stepwise
kk$factors.kept
kk$names.kept

write.csv2((kk$names.kept), file="stepwise.csv")


####################################################################
## Perform random forest stepwise
##################################################################

library(ordinalForest)
stepw_bd_segm_f<-bd_segm_f[,c("id02",
                              "id14",
                              "id16",
                              "id17",
                              "id18",
                              "id24",
                              "id25",
                              "id26",
                              "id27",
                              "id29",
                              "id30",
                              "id34",
                              "id36",
                              "id41",
                              "id42",
                              "id54",
                              "id55",
                              "id57",
                              "id58_1",
                              "id58_2",
                              "id59",
                              "id72",
                              "SINCOSiniestrosMateriales",
                              "SINCOTotalVehiculos",
                              "ContrataDanyosPropios",
                              "DGT_001",
                              "DGT_002",
                              "IGN_001",
                              "ProvinciaID",
                              "Conductor2ID",
                              "Edad",
                              "antiguedad_carnet_años",
                              "id01.Propietario",
                              "id13")]


library(ordinalForest)

require(caTools)
library(caTools)
library(vcd)
samplestep = sample.split(stepw_bd_segm_f$id13, SplitRatio = .70)
trainstep = subset(stepw_bd_segm_f, samplestep == TRUE)
teststep  = subset(stepw_bd_segm_f, samplestep == FALSE)

ordforstep_70 <- ordfor(depvar="id13", data=trainstep, nsets=50, nbest=5, ntreeperdiv=100,
                        ntreefinal=1000)

preds70_step <- predict(ordforstep_70, newdata=teststep)
table(data.frame(true_values=teststep$id13, predictions=preds70_step$ypred))
agreementplot(table(data.frame(true_values=teststep$id13, predictions=preds70_step$ypred)))
estad_bandi_step<-agreementplot(table(data.frame(true_values=teststep$id13, predictions=preds70_step$ypred)))
estad_bandi_step$Bangdiwala_Weighted
#0.6180354
save_incremental()


##################################################################
## Univariate screening
# Univariate ordinal regression
##################################################################

library(pbapply)

#resultados <- pblapply(bd_segm[,-c(1, 3)], function(x){
#  df <- data.frame(y = bd_segm$id13, x=x)
#  tryCatch(clm(y ~ x, data=df), error=function(e) NA)
#})

resultados <- pblapply(bd_segm, function(x){
  df <- data.frame(y = bd_segm$id13, x = x)
  tryCatch(clm(y ~ x, data = df), error = function(e) NA)
})


library(car)
p_values_univariate <- pbsapply(resultados, function(x){
  tryCatch(Anova(x)$`Pr(>Chisq)`, error=function(e) NA)
})

table(p.adjust(p_values_univariate, "fdr") < 0.05)
#Todas son significativas!!!
names(bd_segm)
write.csv2((names(bd_segm)), file="univariate.csv")




##################################################################
#GRADIENT BOOSTING WITH ORDINAL DEPENDENT VARIABLE
##################################################################
install.packages("gbm3", repos = "https://cloud.r-project.org/")

library(gbm)
library(xgboost)
library(Ckmeans.1d.dp)

# Gradient Boosting regression

kkk <- xgb.DMatrix(X, label=bd_segm$id13)
cv <- xgb.cv(data=kkk, nrounds = 200, nthread = 4, nfold = 10)
as.data.frame(cv$evaluation_log)

which.min(as.data.frame(cv$evaluation_log)$test_rmse_mean)
model_gbm <- xgboost(kkk, nrounds = 70)
matriz_importancias <- xgb.importance(model=model_gbm)
xgb.ggplot.importance(matriz_importancias)

plotxgb<-xgb.ggplot.importance(matriz_importancias)
dataplotxgb
dataplotxgb<-plotxgb[[1]]
variables_gboost<-as.data.frame(dataplotxgb[c(1:70),1])

#Variables seleccionadas Gradient Boosting


variables_seleccionadas_gboost <- sapply(names(bd_segm), function(x){
  any(grepl(x, variables_gboost$Feature))
})

names(variables_seleccionadas_gboost[variables_seleccionadas_gboost])

write.csv2(names(variables_seleccionadas_gboost[variables_seleccionadas_gboost]), file="grad_boos.csv")


####################################################################
## Perform random forest gboost
##################################################################
bd_segm$id
write.csv2(bd_segm, file="bd_segm.csv")

library(ordinalForest)
gbost_bd_segm_f<-bd_segm[,c("id01",
                            "id16",
                            "id21",
                            "id27",
                            "id34",
                            "id36",
                            "id37",
                            "id41",
                            "id43",
                            "id44",
                            "id49",
                            "id50",
                            "DGC_001",
                            "DGC_004",
                            "DGC_006",
                            "DGC_007",
                            "DGC_008",
                            "DGC_009",
                            "DGC_012",
                            "DGC_014",
                            "DGC_016",
                            "DGC_018",
                            "DGC_019",
                            "DGC_020",
                            "DGT_002",
                            "IGN_001",
                            "INE_005",
                            "INE_007",
                            "INE_010",
                            "INE_011",
                            "INE_013",
                            "INE_015",
                            "INE_016",
                            "INE_017",
                            "INE_018",
                            "INE_020",
                            "INE_021",
                            "INE_022",
                            "INE_023",
                            "INE_027",
                            "INE_034",
                            "INE_037",
                            "INE_039",
                            "INE_040",
                            "INE_042",
                            "INE_046",
                            "INE_048",
                            "INE_050",
                            "INE_051",
                            "INE_056",
                            "INE_057",
                            "REP_001",
                            "REP_005",
                            "REP_006",
                            "REP_009",
                            "REP_010",
                            "id09_y",
                            "client_days_loyalty",
                            "id17",
                            "id18",
                            "Age",
                            "id50",
                            "id51",
                            "when_to_pay",
                            "id70",
                            "ScoreAda",
                            "SiniestralidadTotalPoliza",
                            "ExposicionTotalPoliza",
                            "ComisionTotalPoliza",
                            "PrimaTotalPoliza",
                            "id13")]



library(caTools)
sample_gb = sample.split(gbost_bd_segm_f$id13, SplitRatio = 0.70)
traingb = subset(gbost_bd_segm_f, sample_gb == TRUE)
testgb  = subset(gbost_bd_segm_f, sample_gb == FALSE)

ordforgb_70 <- ordfor(depvar="id13", data=traingb, nsets=50, nbest=5, ntreeperdiv=100,
                      ntreefinal=1000)
summary(ordforgb_70)

ordforgb70 <- predict(ordforgb_70, newdata=testgb)
table(data.frame(true_values=testgb$id13, predictions=ordforgb70$ypred))

library(vcd)
preds70_gb <- predict(ordforgb_70, newdata=testgb)
table(data.frame(true_values=testgb$id13, predictions=preds70_gb$ypred))
agreementplot(table(data.frame(true_values=testgb$id13, predictions=preds70_gb$ypred)))
estad_bangdi_gb<-agreementplot(table(data.frame(true_values=testgb$id13, predictions=preds70_gb$ypred)))
estad_bangdi_gb$Bangdiwala_Weighted
#0.5627189





save_incremental()

save_incremental()
library(isaves)
