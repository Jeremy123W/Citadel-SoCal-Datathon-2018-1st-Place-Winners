library(data.table)
library(xts)
library(readr)
library(zoo)
library(lubridate)
library(DataAnalytics)
library(moments)
library(readxl)
library(ggplot2)

gc()
chemicals = read_csv("/Users/sophiazhou/Desktop/datathon/Datathon Materials/chemicals.csv")
industry = read_csv("/Users/sophiazhou/Desktop/datathon/Datathon Materials/industry_occupation.csv")
#head(chemicals)
chemicals = as.data.table(chemicals)
industry = as.data.table(industry)
industry=na.omit(industry)
industry[,ave:=mean(total_employed),by='fips']
industry[,ag_ave:=mean(agriculture),by="fips"]
industry[,con_ave:=mean(construction),by='fips']
industry[,man_ave:=mean(manufacturing),by='fips']
industry[,trans_ave:=mean(transport_utilities),by='fips']
industry[,prof_ave:=mean(prof_scientific_waste),by='fips']
industry[,whole_ave:=mean(wholesale_trade),by='fips']
industry[,retail_ave:=mean(retail_trade),by='fips']
industry[,inform_ave:=mean(information),by='fips']
industry[,fin_ins_ave:=mean(finance_insurance_realestate),by='fips']
industry[,edu_ave:=mean(edu_health),by='fips']
industry[,arts_ave:=mean(arts_recreation),by='fips']

industry1=industry[,c('fips','ave','ag_ave','con_ave','man_ave','trans_ave','prof_ave','whole_ave','retail_ave','inform_ave','fin_ins_ave','edu_ave','arts_ave')]
industry1=unique(industry1)

chemicals1=chemicals[year>2009,] #filtered by year to match industry year from 2010-2016
chemicals1=chemicals1[year<2017,]
chemicals1[,ratio:=weighted.mean(value,pop_served),by = c('fips','chemical_species')]
#chemicals1[,ratio:=value/pop_served]
#get diff chemicals
uranium= chemicals1[chemicals1$chemical_species=='Uranium'] #restrict pollutant level
arsenic= chemicals1[chemicals1$chemical_species=='Arsenic']
DEHP= chemicals1[chemicals1$chemical_species=='DEHP']
nitrates= chemicals1[chemicals1$chemical_species=='Nitrates']
acid= chemicals1[chemicals1$chemical_species=='Halo-Acetic Acid']
trihal= chemicals1[chemicals1$chemical_species=='Trihalomethane']

#dt[, c("a", "b")]
uranium=uranium[,c("fips","ratio")]
uranium=unique(uranium)
colnames(uranium)=c('fips','uranium_ratio')

arsenic=arsenic[,c("fips","ratio")]
arsenic=unique(arsenic)
colnames(arsenic)=c('fips','arsenic_ratio')

DEHP=DEHP[,c("fips","ratio")]
DEHP=unique(DEHP)
colnames(DEHP)=c('fips','DEHP_ratio')

nitrates=nitrates[,c("fips","ratio")]
nitrates=unique(nitrates)
colnames(nitrates)=c('fips','nitrates_ratio')

acid=acid[,c("fips","ratio")]
acid=unique(acid)
colnames(acid)=c('fips','acid_ratio')

trihal=trihal[,c("fips","ratio")]
trihal=unique(trihal)
colnames(trihal)=c('fips','trihal_ratio')

data=merge(industry1,nitrates,by='fips')
data1=merge(industry1,acid,by='fips')
data2=merge(industry1,arsenic,by='fips')
data3=merge(industry1,DEHP,by='fips')
data4=merge(industry1,uranium,by='fips')
data5=merge(industry1,trihal,by='fips')

regression=lm(data$nitrates_ratio~data$ave+data$ag_ave+data$con_ave+data$man_ave+data$trans_ave+data$prof_ave+data$whole_ave+data$retail_ave+data$inform_ave+data$fin_ins_ave+data$edu_ave+data$arts_ave)
regression1=lm(data1$acid_ratio~data1$ave+data1$ag_ave+data1$con_ave+data1$man_ave+data1$trans_ave+data1$prof_ave+data1$whole_ave+data1$retail_ave+data1$inform_ave+data1$fin_ins_ave+data1$edu_ave+data1$arts_ave)
regression2=lm(data2$arsenic_ratio~data2$ave+data2$ag_ave+data2$con_ave+data2$man_ave+data2$trans_ave+data2$prof_ave+data2$whole_ave+data2$retail_ave+data2$inform_ave+data2$fin_ins_ave+data2$edu_ave+data2$arts_ave)
regression3=lm(data3$DEHP_ratio~data3$ave+data3$ag_ave+data3$con_ave+data3$man_ave+data3$trans_ave+data3$prof_ave+data3$whole_ave+data3$retail_ave+data3$inform_ave+data3$fin_ins_ave+data3$edu_ave+data3$arts_ave)
regression4=lm(data4$uranium_ratio~data4$ave+data4$ag_ave+data4$con_ave+data4$man_ave+data4$trans_ave+data4$prof_ave+data4$whole_ave+data4$retail_ave+data4$inform_ave+data4$fin_ins_ave+data4$edu_ave+data4$arts_ave)
regression5=lm(data5$trihal_ratio~data5$ave+data5$ag_ave+data5$con_ave+data5$man_ave+data5$trans_ave+data5$prof_ave+data5$whole_ave+data5$retail_ave+data5$inform_ave+data5$fin_ins_ave+data5$edu_ave+data5$arts_ave)

#plot(data$ag_ave,data$nitrates_ratio,xlim=c(0,10000),ylim=c(0,1))
#plot(data$ave,data$trihal_ratio,ylim=c(0,1000000))

#qplot(ag_ave,nitrates_ratio,data=data,col=I("blue"),)+geom_smooth(method="lm",col=I("red"),size=I(1.2),se=FALSE)+theme_bw()
#Winsorize(data$arts_ave)
#data1=data[arts_ave<498782,]
#qplot(arts_ave,nitrates_ratio,data=data,col=I("blue"),)+geom_smooth(method="lm",col=I("red"),size=I(1.2),se=FALSE)+theme_bw()

education = read_csv("/Users/sophiazhou/Desktop/datathon/Datathon Materials/education_attainment.csv")
education=as.data.table(education)
education=education[year==2000,]
education=education[,c('fips','less_than_hs','some_college_or_associates','college_bachelors_or_higher')]

education_merge1=merge(nitrates,education,by='fips')
education_merge2=merge(acid,education,by='fips')
education_merge3=merge(arsenic,education,by='fips')
education_merge4=merge(DEHP,education,by='fips')
education_merge5=merge(uranium,education,by='fips')
education_merge6=merge(trihal,education,by='fips')

education_reg1=lm(education_merge1$nitrates_ratio~education_merge1$less_than_hs+education_merge1$some_college_or_associates+education_merge1$college_bachelors_or_higher)
education_reg2=lm(education_merge2$acid~education_merge2$less_than_hs+education_merge2$some_college_or_associates+education_merge2$college_bachelors_or_higher)
education_reg3=lm(education_merge3$arsenic_ratio~education_merge3$less_than_hs+education_merge3$some_college_or_associates+education_merge3$college_bachelors_or_higher)
education_reg4=lm(education_merge4$DEHP_ratio~education_merge4$less_than_hs+education_merge4$some_college_or_associates+education_merge4$college_bachelors_or_higher)
education_reg5=lm(education_merge5$uranium_ratio~education_merge5$less_than_hs+education_merge5$some_college_or_associates+education_merge5$college_bachelors_or_higher)
education_reg6=lm(education_merge6$trihal_ratio~education_merge6$less_than_hs+education_merge6$some_college_or_associates+education_merge6$college_bachelors_or_higher)


plot(education_merge1$less_than_hs,education_merge1$nitrates_ratio)
qplot(less_than_hs,nitrates_ratio,data=education_merge1,col=I("blue"),xlim=c(0,500000),)+geom_smooth(method="lm",col=I("red"),size=I(1.2),se=FALSE)+theme_bw()
qplot(less_than_hs,acid_ratio,data=education_merge2,col=I("blue"),xlim=c(0,300000),)+geom_smooth(method="lm",col=I("red"),size=I(1.2),se=FALSE)+theme_bw()
qplot(less_than_hs,trihal_ratio,data=education_merge6,col=I("blue"),xlim=c(0,300000),)+geom_smooth(method="lm",col=I("red"),size=I(1.2),se=FALSE)+theme_bw()

