library(twitteR)
library(ROAuth)
library(plyr)
library(stringr)

#setup_twitter_oauth()
#insert in your twitter authentication key

#######################################################
#Extracting from twitter
#######################################################

##########################
#insert profile below:
profile = enfp
#########################

wordVector = rep("empty", length(profile))
for (i in 1:length(profile)){
  print(i)
  tweet = try(userTimeline(profile[i], n = 80, retryOnRateLimit = 10)); #try n = 1 to remove the handles that cause errors.
  if (class(tweet) == "try-error") next;
  #tweet = userTimeline("ardydo", n = 40, retryOnRateLimit=10)

  tweets_cl = tweet
  tweets = sapply(tweets_cl,function(x) x$getText())
  tweets_cl = gsub("(RT|via)((:\\b\\W*@\\w+)+)","",tweets)
  tweets_cl = gsub("http[^[:blank:]]+", "", tweets_cl)
  tweets_cl = gsub("@\\w+", "", tweets_cl)
  tweets_cl <- str_replace_all(tweets_cl,"#[a-z,A-Z]*","")
  tweets_cl = gsub("[ \t]{2,}", "", tweets_cl)
  tweets_cl = gsub("^\\s+|\\s+$", "", tweets_cl)
  tweets_cl <- gsub('\\d+', '', tweets_cl)
  tweets_cl = gsub("[[:punct:]]", ".", tweets_cl)
  tweets_cl = tolower(tweets_cl)
  tweets_collapse = paste(tweets_cl, sep="", collapse=".")

  wordVector[i] = tweets_collapse
}

#change this depending on which category I'm doing

enfpVector = wordVector

##########################
#Save as ____Vector from wordVector
##########################

##################################################################
#Done so far: Each vector represents tweets from each of the 16 categories
##################################################################

estjVector
istjVector
isfjVector
esfjVector

istpVector
isfpVector
estpVector
esfpVector

intjVector
intpVector
entjVector
entpVector

infjVector
infpVector
enfpVector
enfjVector

#########################################################
#Combing vectors and merging data into 1 dataframe
#########################################################

#######################################################
#Analyst
#######################################################

intjA = rep("intj" , 87)
intjB = rep("analyst" , 87)
intjC = rep(1, 87)
intjAB = cbind(intjA, intjB)
intjABC = cbind(intjAB , intjC)
intjABCD = cbind(intjABC, intjVector)

intpA = rep("intp" , 89)
intpB = rep("analyst" , 89)
intpC = rep(1, 89)
intpAB = cbind(intpA, intpB)
intpABC = cbind(intpAB , intpC)
intpABCD = cbind(intpABC, intpVector)

entjA = rep("entj" , 89)
entjB = rep("analyst" , 89)
entjC = rep(1, 89)
entjAB = cbind(entjA, entjB)
entjABC = cbind(entjAB , entjC)
entjABCD = cbind(entjABC, entjVector)

entpA = rep("entp" , 89)
entpB = rep("analyst" , 89)
entpC = rep(1, 89)
entpAB = cbind(entpA, entpB)
entpABC = cbind(entpAB , entpC)
entpABCD = cbind(entpABC, entpVector)

total1 = rbind(intjABCD, intpABCD)
total1 = rbind(total1, entjABCD)
total1 = rbind(total1, entpABCD)

#######################################################
#Diplomat
#######################################################

enfpA = rep("enfp" , 89)
enfpB = rep("diplomat" , 89)
enfpC = rep(2, 89)
enfpAB = cbind(enfpA, enfpB)
enfpABC = cbind(enfpAB , enfpC)
enfpABCD = cbind(enfpABC, enfpVector)

infjA = rep("infj" , 89)
infjB = rep("diplomat" , 89)
infjC = rep(2, 89)
infjAB = cbind(infjA, infjB)
infjABC = cbind(infjAB , infjC)
infjABCD = cbind(infjABC, infjVector)

infpA = rep("infp" , 89)
infpB = rep("diplomat" , 89)
infpC = rep(2, 89)
infpAB = cbind(infpA, infpB)
infpABC = cbind(infpAB , infpC)
infpABCD = cbind(infpABC, infpVector)

enfjA = rep("enfj" , 89)
enfjB = rep("diplomat" , 89)
enfjC = rep(2, 89)
enfjAB = cbind(enfjA, enfjB)
enfjABC = cbind(enfjAB , enfjC)
enfjABCD = cbind(enfjABC, enfjVector)

total2 = rbind(enfpABCD, infpABCD)
total2 = rbind(total2, enfjABCD)
total2 = rbind(total2, infjABCD)

#######################################################
#Sentinel
#######################################################

estjA = rep("estj" , 76)
estjB = rep("sentinel" , 76)
estjC = rep(4, 76)
estjAB = cbind(estjA, estjB)
estjABC = cbind(estjAB , estjC)
estjABCD = cbind(estjABC, estjVector)

istjA = rep("istj" , 84)
istjB = rep("sentinel" , 84)
istjC = rep(4, 84)
istjAB = cbind(istjA, istjB)
istjABC = cbind(istjAB , istjC)
istjABCD = cbind(istjABC, istjVector)

isfjA = rep("isfj" , 85)
isfjB = rep("sentinel" , 85)
isfjC = rep(4, 85)
isfjAB = cbind(isfjA, isfjB)
isfjABC = cbind(isfjAB , isfjC)
isfjABCD = cbind(isfjABC, isfjVector)

esfjA = rep("esfj" , 76)
esfjB = rep("sentinel" , 76)
esfjC = rep(4, 76)
esfjAB = cbind(esfjA, esfjB)
esfjABC = cbind(esfjAB , esfjC)
esfjABCD = cbind(esfjABC, esfjVector)

total4 = rbind(estjABCD, istjABCD)
total4 = rbind(total4, isfjABCD)
total4 = rbind(total4, esfjABCD)

#######################################################
#Explorer
#######################################################

istpA = rep("istp" , 81)
istpB = rep("explorer" , 81)
istpC = rep(3, 81)
istpAB = cbind(istpA, istpB)
istpABC = cbind(istpAB , istpC)
istpABCD = cbind(istpABC, istpVector)

isfpA = rep("isfp" , 88)
isfpB = rep("explorer" , 88)
isfpC = rep(3, 88)
isfpAB = cbind(isfpA, isfpB)
isfpABC = cbind(isfpAB , isfpC)
isfpABCD = cbind(isfpABC, isfpVector)

estpA = rep("estp" , 77)
estpB = rep("explorer" , 77)
estpC = rep(3, 77)
estpAB = cbind(estpA, estpB)
estpABC = cbind(estpAB , estpC)
estpABCD = cbind(estpABC, estpVector)

esfpA = rep("esfp" , 77)
esfpB = rep("explorer" , 77)
esfpC = rep(3, 77)
esfpAB = cbind(esfpA, esfpB)
esfpABC = cbind(esfpAB , esfpC)
esfpABCD = cbind(esfpABC, esfpVector)

total3 = rbind(istpABCD, isfpABCD)
total3 = rbind(total3, estpABCD)
total3 = rbind(total3, esfpABCD)

total = rbind(total1, total2)
total = rbind(total, total3)
total = rbind(total, total4)

total = subset(total, total[,4] != "empty")
str(total)
write.csv(total, file = "punctuation_mbti.csv")

#######################################################

#######################################################
#Personality Categories 
#######################################################

#################
#Sentinels
#################


#################
#Explorer
#################


#################
#Analysts
#################


#################
#Diplomats
#################
