# Install required packages (only once)
install.packages(c("tm", "SnowballC", "wordcloud", "RColorBrewer", "syuzhet", "ggplot2"))

# Load libraries
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(syuzhet)
library(ggplot2)

# Sample text data (no file needed)
text <- c(
  "The team performed exceptionally well and exceeded expectations.",
  "There were some issues in communication and delivery deadlines.",
  "The health and work environment have improved significantly.",
  "I'm happy with the progress and the motivation among the employees.",
  "However, some members are still struggling with time management."
)

# Create a text corpus
TextDoc <- Corpus(VectorSource(text))

# Clean and preprocess text
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
TextDoc <- tm_map(TextDoc, toSpace, "/")
TextDoc <- tm_map(TextDoc, toSpace, "@")
TextDoc <- tm_map(TextDoc, toSpace, "\\|")
TextDoc <- tm_map(TextDoc, content_transformer(tolower))
TextDoc <- tm_map(TextDoc, removeNumbers)
TextDoc <- tm_map(TextDoc, removeWords, stopwords("english"))
TextDoc <- tm_map(TextDoc, removeWords, c("team", "company"))
TextDoc <- tm_map(TextDoc, removePunctuation)
TextDoc <- tm_map(TextDoc, stripWhitespace)
TextDoc <- tm_map(TextDoc, stemDocument)

# Create Term-Document Matrix
TextDoc_dtm <- TermDocumentMatrix(TextDoc)
dtm_m <- as.matrix(TextDoc_dtm)
dtm_v <- sort(rowSums(dtm_m), decreasing = TRUE)
dtm_d <- data.frame(word = names(dtm_v), freq = dtm_v)

# Display top 5 frequent words
head(dtm_d, 5)

# Plot top 5 most frequent words
barplot(dtm_d[1:5,]$freq, las = 2, names.arg = dtm_d[1:5,]$word,
        col = "green", main = "Top 5 most frequent words",
        ylab = "Word frequencies")

# Generate word cloud
set.seed(1234)
wordcloud(words = dtm_d$word, freq = dtm_d$freq, min.freq = 1,
          max.words = 100, random.order = FALSE, rot.per = 0.40,
          colors = brewer.pal(8, "Dark2"))

# Sentiment Analysis
syuzhet_vector <- get_sentiment(text, method = "syuzhet")
bing_vector <- get_sentiment(text, method = "bing")
afinn_vector <- get_sentiment(text, method = "afinn")
nrc_vector <- get_sentiment(text, method = "nrc")

# NRC Sentiment breakdown
nrc_df <- get_nrc_sentiment(text)
td <- data.frame(t(nrc_df))
td_new <- data.frame(rowSums(td))
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new2 <- td_new[1:8, ]

# Plot 1: Sentiment bar plot
ggplot(td_new2, aes(x = sentiment, y = count, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Sentiment Distribution", x = "Sentiment", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot 2: Percentage-based bar plot (base R)
barplot(
  sort(colSums(prop.table(nrc_df[, 1:8]))),
  horiz = TRUE,
  cex.names = 0.7,
  las = 1,
  main = "Emotions in Text",
  xlab = "Percentage"
)
