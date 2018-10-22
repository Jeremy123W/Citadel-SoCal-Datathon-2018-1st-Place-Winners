
library(data.table)
library(ggplot2)
library(randomForest)
library(pscl)
library(corrplot)
library(rsample)      
library(ranger)       
library(caret)       
library(h2o)

waterUsage <- fread("D:\\Datathon 2018\\Datathon Materials\\water_usage.csv", header = T, na.strings = "")
droughts <- fread("D:\\Datathon 2018\\Datathon Materials\\droughts.csv", header = T, na.strings = "")
earnings <- fread("D:\\Datathon 2018\\Datathon Materials\\earnings.csv", header = T, na.strings = "")
education <- fread("D:\\Datathon 2018\\Datathon Materials\\education_attainment.csv", header = T, na.strings = "")
industry_occupation <- fread("D:\\Datathon 2018\\Datathon Materials\\industry_occupation.csv", header = T, na.strings = "")
chemicals <- fread("D:\\Datathon 2018\\Datathon Materials\\chemicals.csv", header = T, na.strings = "")
health <- fread("D:\\Datathon 2018\\Datathon Materials\\county_health.csv", header = T, na.strings = "")

chemicals <- chemicals[,.(polluted=as.numeric(sum(contaminant_level=="Greater than MCL")>0)),by = .(fips,year)]

health$year <- health$Year
health$fips <- health$FIPS
droughts$valid_start <- as.Date(as.character(droughts$valid_start), "%Y-%m-%d")
droughts$valid_end <- as.Date(as.character(droughts$valid_end), "%Y-%m-%d")
droughts$year <- as.numeric(format(droughts$valid_end , "%Y"))

yearly_droughts <- droughts[, .(avg_none = (1-mean(none))/100, avg_d0 = mean(d0)/100, avg_d1 = mean(d1)/100,
                                avg_d2 = mean(d2)/100, avg_d3 = mean(d3)/100, avg_d4 = mean(d4)/100)
                            , by=.(fips, year)]

yearly_droughts$avg_drought <- (yearly_droughts$avg_d2+yearly_droughts$avg_d3+yearly_droughts$avg_d4)>0

tempDT <- merge(yearly_droughts, waterUsage, by = c("fips", "year"))
tempDT <- merge(tempDT, earnings, by = c("fips", "year"))
education$year[which(education$year=='2012-2016')] <- as.numeric(2010)
education$year <- as.numeric(education$year)
tempDT <- merge(tempDT, education, by = c("fips","year"))
tempDT <- merge(tempDT, health, by = c("fips","year"))
tempDT <- merge(tempDT, chemicals, by = c("fips","year"))

tempDT$ground_ratio <- tempDT$gro_wat_3/tempDT$total_withdrawal_3 
tempDT$fresh_ratio <- tempDT$total_withdrawal_1/tempDT$total_withdrawal_3 
tempDT$industry_ratio <- tempDT$ind_9/tempDT$total_withdrawal_3
tempDT$irrigation_ratio <- tempDT$irrigation_3/tempDT$total_withdrawal_3
tempDT$livestock_ratio <- tempDT$livestock_3/tempDT$total_withdrawal_3
tempDT$aqua_ratio <- tempDT$aqua_9/tempDT$total_withdrawal_3
tempDT$mining_ratio <- tempDT$mining_9/tempDT$total_withdrawal_3
tempDT$thermoelectric_ratio <- tempDT$thermoelectric_9/tempDT$total_withdrawal_3
tempDT$dom_per_cap <- as.numeric(tempDT$dom_sup_5)+as.numeric(tempDT$dom_sup_7)

indicators <- as.data.table(cbind(tempDT$ground_ratio,tempDT$fresh_ratio,tempDT$industry_ratio,tempDT$irrigation_ratio
                    ,tempDT$livestock_ratio,tempDT$aqua_ratio,tempDT$mining_ratio,
                    tempDT$thermoelectric_ratio,tempDT$dom_per_cap,tempDT$total_med,tempDT$pct_college_bachelors_or_higher))

colnames(indicators) <- c("ground_ratio","fresh_ratio","industry_ratio","irrigation_ratio","livestock_ratio","aqua_ratio","mining_ratio","thermoelectric_ratio", "domestic_per_capita","earning","education_level")
corrplot(cor(na.omit(indicators)), method = "color", type="upper", tl.col="black", tl.srt=45)

indicators$drought_ratio <- (1-tempDT$avg_none)*100
indicators$drought_ratio <- as.factor(tempDT$avg_drought)
set.seed(123)
rf <- randomForest(drought_ratio~., data = na.omit(indicators), importance = TRUE)
rf$importance

# create training and validation data 
set.seed(123)
valid_split <- initial_split(na.omit(indicators), .8)

# training data
ames_train_v2 <- analysis(valid_split)

# validation data
ames_valid <- as.data.frame(assessment(valid_split))
x_test <- ames_valid[setdiff(names(ames_valid), "drought_ratio")]
y_test <- ames_valid$drought_ratio

rf_oob_comp <- randomForest(
  formula = drought_ratio ~ .,
  data    = ames_train_v2,
  xtest   = x_test,
  ytest   = y_test
)

# extract OOB & validation errors
oob <- rf_oob_comp$err.rate[,3]
validation <- rf_oob_comp$test$err.rate[,3]

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous(labels = scales::dollar) +
  xlab("Number of trees")
















summary(lm(tempDT$avg_none ~ tempDT$ground_ratio))###
summary(lm(tempDT$avg_none ~ tempDT$fresh_ratio))#
summary(lm(tempDT$avg_d0 ~ tempDT$ground_ratio))###
summary(lm(tempDT$avg_d0 ~ tempDT$fresh_ratio))
summary(lm(tempDT$avg_d1 ~ tempDT$ground_ratio))#
summary(lm(tempDT$avg_d1 ~ tempDT$fresh_ratio))
summary(lm(tempDT$avg_d2 ~ tempDT$ground_ratio))###
summary(lm(tempDT$avg_d2 ~ tempDT$fresh_ratio))##
summary(lm(tempDT$avg_d3 ~ tempDT$ground_ratio))###
summary(lm(tempDT$avg_d3 ~ tempDT$fresh_ratio))
summary(lm(tempDT$avg_d4 ~ tempDT$ground_ratio))
summary(lm(tempDT$avg_d4 ~ tempDT$fresh_ratio))
summary(lm(tempDT$avg_none ~ tempDT$industry_ratio))
summary(lm(tempDT$avg_d0 ~ tempDT$industry_ratio))
summary(lm(tempDT$avg_d1 ~ tempDT$industry_ratio))
summary(lm(tempDT$avg_d2 ~ tempDT$industry_ratio))
summary(lm(tempDT$avg_d3 ~ tempDT$industry_ratio))
summary(lm(tempDT$avg_d4 ~ tempDT$industry_ratio))
summary(lm(tempDT$avg_none ~ tempDT$irrigation_ratio))
summary(lm(tempDT$avg_d0 ~ tempDT$irrigation_ratio))
summary(lm(tempDT$avg_d1 ~ tempDT$irrigation_ratio))
summary(lm(tempDT$avg_d2 ~ tempDT$irrigation_ratio))
summary(lm(tempDT$avg_d3 ~ tempDT$irrigation_ratio))
summary(lm(tempDT$avg_d4 ~ tempDT$irrigation_ratio))
summary(lm(tempDT$avg_none ~ tempDT$livestock_ratio))
summary(lm(tempDT$avg_none ~ tempDT$aqua_ratio))
summary(lm(tempDT$avg_none ~ tempDT$mining_ratio))
summary(lm(tempDT$avg_none ~ tempDT$thermoelectric_ratio))
summary(lm(tempDT$avg_none ~ tempDT$total_med))
summary(lm(tempDT$avg_d0 ~ tempDT$total_med))
summary(lm(tempDT$avg_none ~ tempDT$dom_per_cap))
summary(lm(tempDT$avg_d0 ~ tempDT$dom_per_cap))
summary(lm(tempDT$avg_none ~ tempDT$pct_college_bachelors_or_higher))
summary(lm(tempDT$`% Fair/Poor` ~ tempDT$avg_none))
summary(glm(tempDT$polluted~tempDT$avg_none, family = "binomial"))

summary(glm(tempDT$avg_drought ~ tempDT$ground_ratio, family = "binomial"))

ggplot(data=tempDT,aes(`total_med`,`avg_none`))+
  geom_point(lwd=3,color="blue")+ 
  geom_smooth(method='lm',formula=y~x,color="red")

ggplot(data=tempDT,aes(`total_med`,`avg_d0`))+
  geom_point(lwd=3,color="blue")+ 
  geom_smooth(method='lm',formula=y~x,color="red")

ggplot(data=tempDT,aes(`ground_ratio`,`avg_d1`))+
  geom_point(lwd=3,color="blue")+ 
  geom_smooth(method='lm',formula=y~x,color="red")

ggplot(data=tempDT,aes(`ground_ratio`,`avg_d2`))+
  geom_point(lwd=3,color="blue")+ 
  geom_smooth(method='lm',formula=y~x,color="red")

ggplot(data=tempDT,aes(`fresh_ratio`,`avg_d2`))+
  geom_point(lwd=3,color="blue")+ 
  geom_smooth(method='lm',formula=y~x,color="red")

ggplot(data=tempDT,aes(`ground_ratio`,`avg_d3`))+
  geom_point(lwd=3,color="blue")+ 
  geom_smooth(method='lm',formula=y~x,color="red")

ggplot(data=tempDT,aes(`ground_ratio`,`avg_d4`))+
  geom_point(lwd=3,color="blue")+ 
  geom_smooth(method='lm',formula=y~x,color="red")


tempDT <- merge(tempDT, industry_occupation, by = c("fips", "year"))