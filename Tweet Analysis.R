# install.packages("svDialogs")
# install.packages("quanteda")
# install.packages("irlba")
# install.packages("rtweet")
# install.packages("tm")
# install.packages('syuzhet')
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("caTools")
# install.packages("qdap")
# install.packages("lubridate")
# install.packages("e1071")
# install.packages("caret")
# install.packages("rpart.plot")
# install.packages("doSNOW")
# install.packages("wordcloud")
library(wordcloud)
library(doSNOW)
library(rpart.plot)
library(rpart)
library(qdap)
library(stringr)
library(syuzhet)
library(tm)
library(rtweet)
library(svDialogs)
library(ggplot2)
library(dplyr)
library(lubridate)
library(e1071)
library(caret)
library(quanteda)
library(randomForest)
library(caTools)

### Get Twitter API from dev.twitter.com ###
consumer_key <- ''
consumer_secret <- ''
access_token <- ''
access_secret <- ''

token <- create_token(app = "hirshs_app", consumer_key = consumer_key, consumer_secret = consumer_secret,
                      access_token = access_token, access_secret = access_secret)


### Set up dataset ###
account <- dlg_input("Enter a Twitter handle", Sys.info()["user"])$res
tmln <- get_timeline(account, n=5000)
twitterDF <- as.data.frame(tmln)
twitterDF <- twitterDF[twitterDF$is_retweet == F,]
rownames(twitterDF) <- 1:nrow(twitterDF)
twitterDF <- twitterDF[,c(3,5,7,13,14)]

twitterDF$date <- as.Date(twitterDF$created_at)
twitterDF$time <- format(as.POSIXct(twitterDF$created_at), format = "%H:%M:%S")

twitterDF$text <- gsub("&amp;", "&", twitterDF$text)
twitterDF$chars <- nchar(twitterDF$text)
twitterDF$hour <- hour(twitterDF$created)


### Plots ###
twitterDF %>% 
  ggplot(aes(x=hour, fill=screen_name)) + geom_histogram(binwidth = 1)

twitterDF %>% 
  ggplot(aes(x=wday(twitterDF$date, label = T))) + geom_histogram(stat = "count")

twitterDF %>% 
  ggplot(aes(x=favorite_count)) + geom_histogram(binwidth = 5000)

twitterDF %>% 
  ggplot(aes(x=retweet_count)) + geom_histogram(binwidth = 1000)


#conv <- iconv(twitterDF$text, to = "")
tweets_source <- VectorSource(twitterDF$text)
corpus <- Corpus(tweets_source)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(removeWords), stopwords('english'))
#corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)
#corpus <- tm_map(corpus, stemCompletion(corpus))
corpusDF <- data.frame(text = sapply(corpus, as.character), stringsAsFactors = T)

# Create wordcloud
wordcloud(corpus, min.freq = 5, colors = brewer.pal(8, "Dark2"), random.color = T, max.words = 500)


wordDF <- as.vector(twitterDF$text)
emotionDF <- get_nrc_sentiment(wordDF)

sentValue <- get_sentiment(wordDF)
round(prop.table(table(sentValue < 0)),4)

sentType <- ifelse(sentValue < -1, "Negative", 
                   ifelse(sentValue >= -1 & sentValue <= 1, "Neutral",
                          ifelse(sentValue > 1, "Positive", ""
                                 )))

sentDF <- data.frame(sentType, sentValue)
twitterDF$sentValue <- sentValue
twitterDF$sentType <- sentType

sentScore <- data.frame(score = colSums(emotionDF[,]))
sentScore <- cbind("sentiment"=rownames(sentScore), sentScore)
rownames(sentScore) <- 1:nrow(sentScore)

# Plot sentiments
ggplot(sentScore, aes(x=sentiment, y=score)) + geom_bar(aes(fill=sentiment), stat = "identity") +
  theme(legend.position = "none")

table(sentType)
findFreqTerms(dfm, 50)



### Create model to predict Positive or Negative sentiment ###

set.seed(101)
split <- sample.split(twitterDF$text, SplitRatio = 0.7)
train <- subset(twitterDF, split == T)
test <- subset(twitterDF, split == F)

### Data Pre Processing ###
train.tokens <- tokens(train$text, what = "word", remove_numbers = T, remove_punct = T,
                       remove_symbols = T, remove_twitter = T, remove_hyphens = T)

train.tokens <- tokens_tolower(train.tokens) #turn all characters lowercase
train.tokens <- tokens_select(train.tokens, stopwords(), selection = "remove") #remove common words/words with no predictive power
train.tokens <- tokens_wordstem(train.tokens, language = "english") #remove common endings

train.tokens.dfm <- dfm(train.tokens, tolower = F) #turn each word into a column
train.tokens.matrix <- as.matrix(train.tokens.dfm) #convert to matrix

train.tokens.df <- cbind(sentType = train$sentType, convert(train.tokens.dfm, to="data.frame"))
names(train.tokens.df) <- make.names(names(train.tokens.df)) #make sure column names can be read in R
train.tokens.df <- train.tokens.df[,c(1,3:2118)]

set.seed(102)
cv.folds <- createMultiFolds(train$sentType, k=10, times = 3)
cv.ctrl <- trainControl(method = "repeatedcv", number = 10,
                        repeats = 3, index = cv.folds)
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

mod1 <- caret::train(sentType ~., train.tokens.df, method = "rpart",
                     trControl = cv.ctrl, tuneLength = 5)
stopCluster(cl)
mod1