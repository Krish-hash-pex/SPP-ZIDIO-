# SPP-ZIDIO-
Stock Price Prediction Using LSTM
My project develops a hybrid deep-learning system for predicting stock prices of the top
ten Indian equities, combining traditional machine-learning strength with modern sequence 
modeling. Beginning with robust data acquisition, we leverage both public datasets from 
platforms such as Kaggle and real-time feeds via Yahoo Finance and the NSE API to
assemble a comprehensive time-series corpus. The raw data undergoes rigorous 
preprocessing and cleaning through Python libraries like NumPy and Pandas, ensuring 
consistency and handling missing entries after that it goes to K-means Clustering algo
before it flows into our anomaly detection module. In this stage, a Long Short-Term 
Memory (LSTM) network, trained during stable market periods, serves to flag unusual 
patterns; once reconstruction error exceeds predefined thresholds, potential market shocks 
are identified and managed to prevent distorted forecasts.
Building on stable data, our system employs Temporal Convolutional Networks (TCNs) to 
capture both short-term momentum and long-range trends in stock movements. These 
convolutional encoders transform raw price sequences into compact embeddings,
effectively summarizing market dynamics in a form amenable to downstream learning. For 
our core regression engine, we replace traditional gradient-boosting implementations with 
LightGBM, capitalizing on its histogram-based tree growth and native handling of 
categorical features to achieve faster training times and competitive accuracy. To promote
transparency, we integrate SHAP-based explainability, which illuminates feature 
contributions and empowers users to understand why the model favors certain signals over 
others. By structuring our pipeline in modular stages—data ingestion, anomaly filtering, 
embedding extraction, and boosted regression—we maintain clarity, ease of maintenance, 
and the potential for rapid iteration.
For user interaction and deployment, we adopt a full-Python stack that balances simplicity 
with scalability. The frontend interface is built with Streamlit, allowing us to craft intuitive 
dashboards and input controls using minimal code, while the backend prediction service is 
exposed through FastAPI, offering asynchronous, high-performance REST endpoints and 
auto-generated documentation via Pydantic. This separation of concerns ensures that our 
prediction logic can be tested independently of the UI, and that new features—such as 
continuous online learning to adapt to concept drift or integration of alternative models 
like Random Forest, SVR, or KNN—can be slotted in with minimal disruption. The 
resulting proof-of-concept not only serves as a demonstrable base model for academic 
evaluation but also lays a clear roadmap for future.

