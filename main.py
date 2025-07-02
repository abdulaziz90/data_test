import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_FILE = 'DS_task_data.xlsb'


def load_data():
    customers = pd.read_excel(DATA_FILE, sheet_name='customers', engine='pyxlsb')
    bills = pd.read_excel(DATA_FILE, sheet_name='bills', engine='pyxlsb')
    origin = pd.Timestamp('1899-12-30')
    for col in ['invoice_date', 'due_date', 'settled_date']:
        bills[col] = origin + pd.to_timedelta(bills[col], unit='D')
    return customers, bills


def generate_features(customers: pd.DataFrame, bills: pd.DataFrame) -> pd.DataFrame:
    bills = bills.copy()
    bills['settled_date_filled'] = bills['settled_date'].fillna(pd.Timestamp.now())
    bills['days_to_pay'] = (bills['settled_date_filled'] - bills['invoice_date']).dt.days
    bills['days_overdue'] = (pd.Timestamp.now() - bills['due_date']).dt.days
    bills['overdue_60'] = bills['days_overdue'] > 60
    bills['aged_balance'] = bills['balance'].where(bills['overdue_60'], 0)

    agg_funcs = {
        'amount': ['sum', 'mean'],
        'amount_settled': 'sum',
        'days_to_pay': 'mean',
        'balance': 'sum',
        'aged_balance': 'sum'
    }
    agg = bills.groupby('customer_account').agg(agg_funcs)
    agg.columns = [
        'total_amount',
        'mean_invoice',
        'total_paid',
        'mean_delay',
        'open_balance',
        'aged_balance'
    ]

    def slope(series):
        if len(series) < 2:
            return 0.0
        idx = series.index
        x = bills.loc[idx, 'invoice_date'].map(pd.Timestamp.toordinal)
        y = series.values
        return np.polyfit(x, y, 1)[0]

    trend = bills.groupby('customer_account')['days_to_pay'].apply(slope)
    agg['payment_trend'] = trend

    features = customers.merge(agg, left_on='customer_account', right_index=True, how='left')
    features.fillna({
        'total_amount': 0,
        'mean_invoice': 0,
        'total_paid': 0,
        'mean_delay': 0,
        'open_balance': 0,
        'aged_balance': 0,
        'payment_trend': 0
    }, inplace=True)

    features['high_risk'] = (features['aged_balance'] > 0).astype(int)
    return features


def trend_analysis(bills: pd.DataFrame):
    monthly = bills.set_index('invoice_date')['amount'].resample('M').sum()
    plt.figure(figsize=(10, 4))
    monthly.plot(title='Monthly Invoiced Amounts')
    plt.ylabel('Amount')
    plt.tight_layout()
    plt.savefig('monthly_trend.png')


def distribution_visuals(customers: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    sns.countplot(x='internal_credit_rating', data=customers)
    plt.title('Distribution of Internal Credit Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rating_distribution.png')


def assess_credit_rating(features: pd.DataFrame):
    summary = features.groupby('internal_credit_rating')['aged_balance'].mean()
    plt.figure(figsize=(8, 4))
    summary.plot(kind='bar')
    plt.ylabel('Average Aged Debt')
    plt.title('Credit Rating vs Aged Debt')
    plt.tight_layout()
    plt.savefig('credit_rating_vs_aged_debt.png')
    print('Average aged debt by rating:')
    print(summary)


def build_model(features: pd.DataFrame):
    numeric = [
        'experian_credit_score',
        'count_sites',
        'contracted_annual_volume',
        'total_amount',
        'total_paid',
        'mean_invoice',
        'mean_delay',
        'open_balance',
        'aged_balance',
        'payment_trend'
    ]

    categorical = [
        'sme_mbs',
        'segment_id',
        'cust_group',
        'payment_method',
        'internal_credit_rating',
        'sales_route',
        'segment',
        'tenure_group'
    ]

    X = features[numeric + categorical].copy()
    y = features['high_risk']
    X[numeric] = X[numeric].fillna(0)
    X[categorical] = X[categorical].fillna('Unknown')

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    clf = Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


def main():
    customers, bills = load_data()
    features = generate_features(customers, bills)

    trend_analysis(bills)
    distribution_visuals(customers)
    assess_credit_rating(features)
    build_model(features)


if __name__ == '__main__':
    main()
