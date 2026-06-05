from pydantic import BaseModel


class PredictionRequest(BaseModel):

    TransactionId: str
    BatchId: str

    AccountId: str
    SubscriptionId: str
    CustomerId: str

    CurrencyCode: str
    ProviderId: str

    ProductCategory: str
    ChannelId: str

    Amount: float
    Value: float

    TransactionStartTime: str

    PricingStrategy: int

class PredictionResponse(BaseModel):

    risk_probability: float
    prediction: str