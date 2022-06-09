#透過dplyr連接資料庫(參考網頁:[https://waylan25.blogspot.com/2017/01/r-database.html])
library(dplyr)
library(DBI)
library(RSQLite)
#連接資料庫
setwd('C:/Users/ivers/Desktop')
db  = dbConnect(dbDriver("SQLite"),"final.sqlite")
#查看資料表
dbListTables(db)
#讀取資料表中的資料
db_original<-as.data.frame(dbReadTable(db, "original"))
lm1_coef<-as.data.frame(dbReadTable(db, "lm1_coef"))
lm2_coef<-as.data.frame(dbReadTable(db, "lm2_coef"))
rf_1_<-as.data.frame(dbReadTable(db, "rf_1_"))
gbm_1<-as.data.frame(dbReadTable(db, "gbm_1"))
rf_2_<-as.data.frame(dbReadTable(db, "rf_2_"))
gbm_2<-as.data.frame(dbReadTable(db, "gbm_2"))
df_lm1<-as.data.frame(dbReadTable(db, "df_lm1"))
df_lm2<-as.data.frame(dbReadTable(db, "df_lm2"))
dfrf_1<-as.data.frame(dbReadTable(db, "dfrf_1"))
dfrf_2<-as.data.frame(dbReadTable(db, "dfrf_2"))
dfgbm_1<-as.data.frame(dbReadTable(db, "dfgbm_1"))
dfgbm_2<-as.data.frame(dbReadTable(db, "dfgbm_2"))
db_1<-dbReadTable(db, "test")
a <-as.data.frame(c(1:1338))
colnames(a)<-c('new_a')
db_original_1<-db_original
db_original_1<-cbind(db_original,a)
#斷開資料庫連接:
dbDisconnect(db)
db_n<-as.data.frame(apply(db_1[,1:7],2,as.integer))
#資料標準化:
db_1_6<-scale(db_n[1:6])
new_db<-as.data.frame(cbind(db_1_6,db_n[,7]))
#視覺化:
library(gridExtra)
library(ggplot2)
library(ggthemes)
db_1<-as.data.frame(db_1)
db_original_1$age<-as.integer(db_original_1$age)
db_original_1$bmi<-as.integer(db_original_1$bmi)
db_original_1$charges<-as.integer(db_original_1$charges)

###################
colnames(db_original_1)
df<-data.frame(c("sex","smoker","region"),
               c("male=1; female=0","yes=1; no=0","西南=0; 東南=1; 西北=2; 東北=3"))
colnames(df)<-c("variable","transform")

df_<-data.frame(c("smoker","children","bmi","age","region","sex"),
                c("有抽菸比較貴","越多越貴","越高越貴","越大越貴",
                  "東北 < 西北 < 東南 < 西南","女生相對男生貴"))
colnames(df_)<-c("variable","與費用的關係")
###################

unique(db_original_1$sex)
sum(db_original_1$sex=="female")
sum(db_original_1$sex=="male")
db_original_1_sex<-data.frame(c("female","male"),c(662,676))
colnames(db_original_1_sex)<-c("sex","count")

unique(db_original_1$children)
sum(db_original_1$children=="0")
sum(db_original_1$children=="1")
sum(db_original_1$children=="2")
sum(db_original_1$children=="3")
sum(db_original_1$children=="4")
sum(db_original_1$children=="5")
db_original_1_children<-data.frame(c("0","1","2","3","4","5"),c(574,324,240,157,25,18))
colnames(db_original_1_children)<-c("children","count")

unique(db_original_1$smoker)
sum(db_original_1$smoker=="yes")
sum(db_original_1$smoker=="no")
db_original_1_smoker<-data.frame(c("yes","no"),c(274,1064))
colnames(db_original_1_smoker)<-c("smoker","count")

unique(db_original_1$region)
sum(db_original_1$region=="southwest")
sum(db_original_1$region=="southeast")
sum(db_original_1$region=="northwest")
sum(db_original_1$region=="northeast")
db_original_1_region<-data.frame(c("southwest","southeast","northwest","northeast"),c(325,364,325,324))
colnames(db_original_1_region)<-c("region","count")

g1<-ggplot(db_original_1, aes(x=age)) +
  geom_histogram(bins = 20,color="black",fill="blue",alpha=0.5)+
  theme_economist()+
  labs(x="age",y="數量")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
g2<-ggplot(db_original_1, aes(x=bmi)) +
  geom_histogram(bins = 20,color="black",fill="blue",alpha=0.5)+
  theme_economist()+
  labs(x="bmi",y="數量")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
g3<-ggplot(db_original_1_sex,aes(x=sex,y=count,fill=sex)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=count), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  labs(x="sex",y="數量")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
g4<-ggplot(db_original_1_children,aes(x=children,y=count,fill=children)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=count), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  labs(x="children",y="數量")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
g5<-ggplot(db_original_1_smoker,aes(x=smoker,y=count,fill=smoker)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=count), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  labs(x="smoker",y="數量")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
g6<-ggplot(db_original_1_region,aes(x=region,y=count,fill=region)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=count), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  labs(x="region",y="數量")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
g7<-ggplot(db_original_1, aes(x=charges)) +
  geom_histogram(bins = 20,color="black",fill="blue",alpha=0.5,)+
  theme_economist()+
  labs(x="charges",y="數量")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
grid.arrange(g1,g2,g3,g4,g5,g6,g7, nrow=2, ncol=4)
####################################################
#1
library(scales)
c1<-ggplot(lm1_coef,aes(x=variable,y=relative_importance_1,fill=variable)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=relative_importance_1), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
c2<-ggplot(rf_1_,aes(x=variable,y=percentage,fill=variable)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=label_percent()(percentage)), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
c3<-ggplot(gbm_1,aes(x=variable,y=percentage,fill=variable)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=label_percent()(percentage)), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
grid.arrange(c1,c2,c3, nrow=1, ncol=3)
#2
c_1<-ggplot(lm2_coef,aes(x=variable,y=relative_importance_2,fill=variable)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=relative_importance_2), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
c_2<-ggplot(rf_2_,aes(x=variable,y=percentage,fill=variable)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=label_percent()(percentage)), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
c_3<-ggplot(gbm_2,aes(x=variable,y=percentage,fill=variable)) +
  geom_bar(color="black",stat="identity", alpha=0.5)+
  geom_text(aes(label=label_percent()(percentage)), vjust=1.6, size=4)+
  theme_economist()+
  theme(legend.position="right")+
  theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
grid.arrange(c_1,c_2,c_3, nrow=1, ncol=3)

##################################################
#建立dashboard
library(shinydashboard)
library(shinythemes)
library(shiny)
library(data.table)
library(DT)

ui<-dashboardPage(
  dashboardHeader(title="專題報告"),
  #第一頁 
  dashboardSidebar( 
    sidebarMenu( 
      menuItem("專題資料", tabName = "id_1", 
               icon = icon("dashboard")),
      menuItem("模型選擇", icon = icon("bar-chart-o"), startExpanded = TRUE,#將模型選項縮排
               menuSubItem("使用全部變數的模型", tabName = "id_2", 
                           icon = icon("hand-point-right")),
               menuSubItem("挑選過變數的模型", tabName = "id_3", 
                           icon = icon("hand-point-right")),
               menuSubItem("自變數與應變數的關係", tabName = "id_4", 
                           icon = icon("hand-point-right"))
      )
    )
  ),
  dashboardBody(
    #第二頁
    tabItems( 
      tabItem(tabName = "id_1", 
              fluidRow(
                box(width = 7,
                    height = 530,
                    status = "primary",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    dataTableOutput("db_original"),
                    downloadButton("download", "Download .csv")),
                box(width = 5,
                    height = 530,
                    status = "warning",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    plotOutput("plot_original_1")
                ),
                box(width = 7,
                    height = 250,
                    titlePanel(tags$b("資料描述")),
                    status = "success",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    h3(
                      tags$div("通過年齡、性別、bmi、扶養兒童人數、是否吸煙、地區等變數，",
                               tags$br(),
                               "進行分析與建模，並比較模型的各項指標，",
                               tags$br(),
                               "選出誤差最小、準確度最高的模型進行預測，",
                               tags$br(),
                               "最後比較出各項變數對預測費用的影響程度。"))
                ),
                box(width = 5, 
                    height = 250,
                    status = "warning",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    plotOutput("plot_original_2")
                )
              )),
      tabItem(tabName = "id_2",
              fluidPage(
                box(width=12,
                    height = 400,
                    titlePanel(tags$b("全部變數的影響程度")),
                    status = "warning",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    plotOutput("c"))
              ),
              fluidPage(
                box(width=4,
                    height = 300,
                    titlePanel(tags$b("LinearRegression_1模型評估")),
                    status = "success",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    div(tableOutput("c_all_1"),style = "font-size:150%")),
                box(width=4,
                    height = 300,
                    titlePanel(tags$b("H2O-RandomForest_1模型評估")),
                    status = "primary",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    div(tableOutput("c_all_2"),style = "font-size:150%")),
                box(width=4,
                    height = 300,
                    titlePanel(tags$b("H2O-GBM_1模型評估")),
                    status = "danger",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    div(tableOutput("c_all_3"),style = "font-size:150%"))
              )
      ),
      tabItem(tabName = "id_3",
              fluidPage(
                box(width=12,
                    height = 400,
                    titlePanel(tags$b("挑選過變數的影響程度")),
                    status = "warning",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    plotOutput("c_"))
              ),
              fluidPage(
                box(width=4,
                    height = 300,
                    titlePanel(tags$b("LinearRegression_2模型評估")),
                    status = "success",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    div(tableOutput("c_part_1"),style = "font-size:150%")),
                box(width=4,
                    height = 300,
                    titlePanel(tags$b("H2O-RandomForest_2模型評估")),
                    status = "primary",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    div(tableOutput("c_part_2"),style = "font-size:150%")),
                box(width=4,
                    height = 300,
                    titlePanel(tags$b("H2O-GBM_2模型評估")),
                    status = "danger",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    div(tableOutput("c_part_3"),style = "font-size:150%"))
              )),
      tabItem(tabName = "id_4",
              fluidPage(
                box(width=10,
                    titlePanel(tags$b(
                      sprintf("回歸方程式 :\n[charges]=24224.32[smoker]+536.79[children]\n+314.56[bmi]+246.85[age]-168.22[region]-317.48[sex]-10919.65"))),
                    status = "warning",
                    solidHeader = TRUE,
                    collapsible = TRUE,
                ),
                box(width=5,
                    dataTableOutput("df"),
                    status = "success",
                    solidHeader = TRUE,
                    collapsible = TRUE,),
                box(width=5,
                    dataTableOutput("df_"),
                    status = "primary",
                    solidHeader = TRUE,
                    collapsible = TRUE,)
                
              )
      )
    )
  )
)

server<-function(input,output){
  #第一頁
  output$db_original <- renderDataTable(db_original,options = list(scrollX = TRUE))
  output$download <- downloadHandler(
    filename = function() {
      paste0("醫療費用資料表", ".csv")
    },
    content = function(file) {
      vroom::vroom_write(data(), file)
    }
  )
  output$plot_original_1 <-renderPlot(grid.arrange(g3,g4,g5,g6, nrow=2, ncol=2),height = 495)
  output$plot_original_2 <-renderPlot(grid.arrange(g1,g2,g7, nrow=1, ncol=3),height = 210)
  #第二頁
  output$c<-renderPlot(
    grid.arrange(c1,c2,c3, nrow=1, ncol=3),height = 300)
  output$c_all_1<-renderTable(df_lm1,width = 400,height = 300)
  output$c_all_2<-renderTable(dfrf_1,width = 400,height = 300)
  output$c_all_3<-renderTable(dfgbm_1,width = 400,height = 300)
  #第三頁
  output$c_<-renderPlot(
    grid.arrange(c_1,c_2,c_3, nrow=1, ncol=3),height = 300)
  output$c_part_1<-renderTable(df_lm2,width = 400,height = 300)
  output$c_part_2<-renderTable(dfrf_2,width = 400,height = 300)
  output$c_part_3<-renderTable(dfgbm_2,width = 400,height = 300)
  output$df<-renderDataTable(df,width = 400,height = 300)
  output$df_<-renderDataTable(df_,width = 400,height = 300)
}

shinyApp(ui = ui, server = server)
