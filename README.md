# MODERN: Image-based furniture recommender

# Background

I’m sure all of at one point shopped for furnitures both online and offline and have been swarmed by the swear amount of furnitures that are out there. Online retailers such as Ikea or Amazon offers similar array of products, chosen by product recommender systems. However, when shopping offline, we often don’t have the luxury to find alternatives or similar products across different websites and I wanted to make this search process a little bit easier. 

This led me to wonder if AI-powered content I’m sure all of at one point shopped for furnitures both online and offline and have been swarmed by the swear amount of furnitures that are out there. Online retailers such as Ikea or Amazon offers similar array of products, chosen by product recommender systems. However, when shopping offline, we often don’t have the luxury to find alternatives or similar products across different websites and I wanted to make this search process a little bit easier. This led me to wonder if AI-powered content recommendation could help customers find the product that they want that incorporates style, texture and color. In doing so, we can save both customers and retailers valuable time in discovering the products to buy and sell.

# Goal

I propose an application which takes in a product image, analyzes its design features using convolutional neural network, and recommends products in categories of choice with similar style elements.

# Methodologies

  1. Collect 4000 images of various different living room furnitures
  2. Utilize Convolutional Neural Networks with transfer learning to modify VGG16, which is a pre-trained model, to perform a multi-label classification task that tries to classify the category of furniture an input image belongs to. This is done by replacing the last three dense layers of VGG16 with new dense layers and retrained the network with my image library. 
  3. To gain insights from how layers of the convolutional neural network represent an image and how we can use this computed representation, utilize the feature maps as feature extractors.
  4. The feature maps are used to compute gram matrices, which basically measures degree of correlations between feature maps which will act as a measure of style itself. This will highlight the most salient features that best represent the furniture. We can then build a design feature library of these gram matrice and when a user provides a image of a new furniture, a design feature vector is encoded in a similiar manner. 
  5. The feature maps are used to compute gram matrices, which basically measures degree of correlations between feature maps which will act as a measure of style itself. We can then build a design feature library of these gram matrice and when a user provides a image of a new furniture, a design feature vector is encoded in a similar manner. 
  6. The design feature vector is used to calculate L2 norm as similiarity search metrics and the furniture with the closest Euclidean distance are returned as recommendations. 

# Credits

## Codes

This project utilizes 
  - Austin McKay's [Style Stack Repo](https://github.com/TheAustinator/style-stack)



