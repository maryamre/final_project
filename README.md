final_project, data inference from customers forum posts

Overview

For tech companies, specially the ones with subscription based business models, users are constantly providing valuable inputs for the products in the forums through their posts. To make the best use of the provided inputs and improve user experience, this project performs natural language processing (NLP) on the forum texts to find the most important issues that users are struggling with and finds the optimal number of topics for each product. Using this information, appropriate interventions can be performed to improve different aspects of the products where users have issue with. Moreover, post recommendation are suggested to be added to the forum to avoid repeated posts.

Analysis

To extract the dominant topics, I created a text preprocessing-cleaning-tokenizing-TFIDF-NMF pipeline. number of replies and number of views are added as extra features in graph lab to magnify these effects. Separation and cohesion metrics were used to find the optimal number of topic for each product. Using the coefficients from matrix factorization along with TFIDF, I found the most relevant post containing a specific key word. And finally, recommendation engine is created to show most similar previous posts to a question that a user has to ask.

Validating the model with domain knowledge

Subject matter expert(SME) with the domain knowledge verified the results from the analysis of this project.
