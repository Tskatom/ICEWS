library("vars")
load_data <- function(filename){
  icews_data <- read.table(filename, header=TRUE, sep=',')
  return(icews_data)
}

setwd("/home/weiw/workspace/ICEWS/src/experiment/VAR/")
icews_exp <- function(region, event)
{
  icews_file <- sprintf("./data/%s_mena_%s.csv", region, event)
  icews_data <- load_data(icews_file)
  new_icews_data <- icews_data[-1]
  total_count <- nrow(new_icews_data)
  test_period <- 12
  testY <- new_icews_data[(total_count-test_period+1):total_count,]
  preds <- c()
  for(i in seq(test_period,1,-1))
  {
    var1 <- VAR(new_icews_data[1:(total_count-i),])
    pred <- predict(var1, n.ahead=1)
    pred_count <- c()
    for(fcst in pred$fcst)
    {
      p <- fcst[1]
      pred_count <- c(pred_count, p)
    }
    preds <- c(preds, pred_count)
  }
  preds <- floor(preds)
  preds[preds < 0] = 0
  final_preds <- matrix(preds, nrow=test_period, ncol=dim(new_icews_data)[2], byrow=TRUE)
  colnames(final_preds) <- colnames(new_icews_data)
  
  write.table(testY, file=sprintf('./data/%s_testY_%s.csv', region, event), sep=',', quote=TRUE, row.names=FALSE)
  write.table(final_preds, file=sprintf('./data/%s_predictions_%s.csv', region, event), sep=',', quote=TRUE, row.names=FALSE)
}

icews_exp("city", "14")
icews_exp("country", "14")

icews_exp("city", "17")
icews_exp("country", "17")

icews_exp("city", "18")
icews_exp("country", "18")


