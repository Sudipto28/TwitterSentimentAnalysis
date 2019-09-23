from TwitterSentimentAnalysis import sentiment_module

statement1 = 'The movie was worst. It was awful experience'
statement2 = 'It was the good. But could have been better.'
statement3 = 'This is the best movie I have watched in ages. 10/10.'

print('Statement1: ', sentiment_module.sentiment(statement1))
print('Statement2: ', sentiment_module.sentiment(statement2))
print('Statement3: ', sentiment_module.sentiment(statement3))

