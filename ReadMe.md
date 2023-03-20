Flights are known to be the safest mode of transportation. However, though the probability of a crash is low, the damages and casualty rate in the event of an accident are significantly high. Air Traffic Control systems currently record the real-time positions of flights, but is that enough?
What if the connection between the flight and the air traffic control is lost? 

With trajectory prediction embedded with mapping, we can accurately visualise where the flight may be heading.
Our solution is to provide a real-time trajectory mapping and prediction system for domestic flights across the top 20 busiest airports in India. Under the domain of AI, specifically Time Series Data Analysis, we have implemented an LSTM (long short-term memory) deep learning neural network to predict the latitude, longitude, and altitude of an entire flight path.
The user is allowed to select any airport for origin and destination, for which if sufficient flight data is available between the two airports, our program web scrapes the trajectory data from an open data source for the five most recent flights. Then, our LSTM model is being trained on the first four flight paths, with constant validation after each epoch, and is finally tested on the fifth flight path.

We constructed our LSTM model with 128 neurons, given the complexity of data, with a dropout of 20% to avoid overfitting and to generalise our model. With 15 epochs and a batch size of 32, we managed to significantly reduce the train and test loss to below 1% and 0.1% respectively.
Our predictions were astonishingly similar to the actual trajectories, showcasing how efficient LSTM is in forecasting future positions based on recurrent patterns in a recorded time series.

Yet, we took our project one step further. With a python program to convert the co-ordinates to a KML file, we decided to allow the user to actually visualise the entire flight from the cockpit view in Google Earth 3D. This proves to be extremely helpful in two scenarios:

1.	It can be used as a flight simulator for pilots, giving them a clear picture of where the runways are located and when to make a significant turn. It also acts as a model for the most optimum flight path that can be taken.

2.	In the event of a crash and connection loss, this model can be used to see the environment around the flight at the last known co-ordinate, helping government officials to envision natural landmarks that exist at that point, if in case the flight may have hit a mountain or tall structure. The future trajectory predictions can also be used to more accurately calculate where the flight may have crash landed, allowing emergency services to reach the crash site faster, which could potentially save more lives.

Our goal is to make flights as safe as possible, and we would love to develop this project further in collaboration with the government. The best part about our solution is its scalability. This forecasting system can be expanded to all flights, not just in India but internationally as well!

So let us work together and revolutionise the aviation industry!
