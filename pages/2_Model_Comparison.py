import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")
st.markdown("### TF-IDF + Logistic Regression vs Fine-tuned DistilBERT")
st.divider()

# Overall Metrics
st.markdown("## Overall Metrics")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### TF-IDF + Logistic Regression")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "86%")
    c2.metric("Macro F1", "0.86")
    c3.metric("Weighted F1", "0.86")

with col2:
    st.markdown("#### DistilBERT")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "91%")
    c2.metric("Macro F1", "0.91")
    c3.metric("Weighted F1", "0.91")

st.divider()

# Per class F1
st.markdown("## Per-Class F1 Score Comparison")

f1_data = {'TF-IDF F1': {'activate_my_card': 0.97, 'age_limit': 1.0, 'apple_pay_or_google_pay': 0.99, 'atm_support': 0.83, 'automatic_top_up': 0.95, 'balance_not_updated_after_bank_transfer': 0.7, 'balance_not_updated_after_cheque_or_cash_deposit': 0.88, 'beneficiary_not_allowed': 0.89, 'cancel_transfer': 0.92, 'card_about_to_expire': 0.95, 'card_acceptance': 0.78, 'card_arrival': 0.8, 'card_delivery_estimate': 0.89, 'card_linking': 0.92, 'card_not_working': 0.69, 'card_payment_fee_charged': 0.84, 'card_payment_not_recognised': 0.73, 'card_payment_wrong_exchange_rate': 0.9, 'card_swallowed': 0.86, 'cash_withdrawal_charge': 0.88, 'cash_withdrawal_not_recognised': 0.8, 'change_pin': 0.96, 'compromised_card': 0.78, 'contactless_not_working': 0.86, 'country_support': 0.9, 'declined_card_payment': 0.81, 'declined_cash_withdrawal': 0.83, 'declined_transfer': 0.91, 'direct_debit_payment_not_recognised': 0.88, 'disposable_card_limits': 0.8, 'edit_personal_details': 0.99, 'exchange_charge': 0.86, 'exchange_rate': 0.91, 'exchange_via_app': 0.84, 'extra_charge_on_statement': 0.77, 'failed_transfer': 0.8, 'fiat_currency_support': 0.83, 'get_disposable_virtual_card': 0.74, 'get_physical_card': 0.88, 'getting_spare_card': 0.9, 'getting_virtual_card': 0.78, 'lost_or_stolen_card': 0.85, 'lost_or_stolen_phone': 0.99, 'order_physical_card': 0.82, 'passcode_forgotten': 1.0, 'pending_card_payment': 0.86, 'pending_cash_withdrawal': 0.95, 'pending_top_up': 0.86, 'pending_transfer': 0.71, 'pin_blocked': 0.83, 'receiving_money': 0.92, 'Refund_not_showing_up': 0.92, 'request_refund': 0.83, 'reverted_card_payment?': 0.84, 'supported_cards_and_currencies': 0.89, 'terminate_account': 0.95, 'top_up_by_bank_transfer_charge': 0.79, 'top_up_by_card_charge': 0.94, 'top_up_by_cash_or_cheque': 0.86, 'top_up_failed': 0.77, 'top_up_limits': 0.92, 'top_up_reverted': 0.82, 'topping_up_by_card': 0.78, 'transaction_charged_twice': 0.95, 'transfer_fee_charged': 0.88, 'transfer_into_account': 0.82, 'transfer_not_received_by_recipient': 0.8, 'transfer_timing': 0.81, 'unable_to_verify_identity': 0.83, 'verify_my_identity': 0.76, 'verify_source_of_funds': 0.94, 'verify_top_up': 1.0, 'virtual_card_not_working': 0.55, 'visa_or_mastercard': 0.95, 'why_verify_identity': 0.82, 'wrong_amount_of_cash_received': 0.9, 'wrong_exchange_rate_for_cash_withdrawal': 0.76}, 'DistilBERT F1': {'activate_my_card': 0.98, 'age_limit': 0.98, 'apple_pay_or_google_pay': 1.0, 'atm_support': 0.98, 'automatic_top_up': 0.94, 'balance_not_updated_after_bank_transfer': 0.78, 'balance_not_updated_after_cheque_or_cash_deposit': 0.96, 'beneficiary_not_allowed': 0.9, 'cancel_transfer': 0.95, 'card_about_to_expire': 0.99, 'card_acceptance': 0.91, 'card_arrival': 0.81, 'card_delivery_estimate': 0.81, 'card_linking': 0.95, 'card_not_working': 0.9, 'card_payment_fee_charged': 0.86, 'card_payment_not_recognised': 0.86, 'card_payment_wrong_exchange_rate': 0.9, 'card_swallowed': 0.93, 'cash_withdrawal_charge': 0.94, 'cash_withdrawal_not_recognised': 0.87, 'change_pin': 0.98, 'compromised_card': 0.89, 'contactless_not_working': 0.95, 'country_support': 0.95, 'declined_card_payment': 0.85, 'declined_cash_withdrawal': 0.94, 'declined_transfer': 0.84, 'direct_debit_payment_not_recognised': 0.89, 'disposable_card_limits': 0.91, 'edit_personal_details': 0.98, 'exchange_charge': 0.95, 'exchange_rate': 0.94, 'exchange_via_app': 0.88, 'extra_charge_on_statement': 0.96, 'failed_transfer': 0.87, 'fiat_currency_support': 0.87, 'get_disposable_virtual_card': 0.87, 'get_physical_card': 0.96, 'getting_spare_card': 0.94, 'getting_virtual_card': 0.84, 'lost_or_stolen_card': 0.89, 'lost_or_stolen_phone': 0.96, 'order_physical_card': 0.88, 'passcode_forgotten': 1.0, 'pending_card_payment': 0.94, 'pending_cash_withdrawal': 0.97, 'pending_top_up': 0.89, 'pending_transfer': 0.84, 'pin_blocked': 0.92, 'receiving_money': 0.92, 'Refund_not_showing_up': 0.93, 'request_refund': 0.93, 'reverted_card_payment?': 0.82, 'supported_cards_and_currencies': 0.91, 'terminate_account': 0.99, 'top_up_by_bank_transfer_charge': 0.88, 'top_up_by_card_charge': 0.92, 'top_up_by_cash_or_cheque': 0.92, 'top_up_failed': 0.85, 'top_up_limits': 0.98, 'top_up_reverted': 0.83, 'topping_up_by_card': 0.78, 'transaction_charged_twice': 0.98, 'transfer_fee_charged': 0.89, 'transfer_into_account': 0.83, 'transfer_not_received_by_recipient': 0.81, 'transfer_timing': 0.85, 'unable_to_verify_identity': 0.91, 'verify_my_identity': 0.77, 'verify_source_of_funds': 1.0, 'verify_top_up': 1.0, 'virtual_card_not_working': 0.87, 'visa_or_mastercard': 0.96, 'why_verify_identity': 0.68, 'wrong_amount_of_cash_received': 0.94, 'wrong_exchange_rate_for_cash_withdrawal': 0.89}}

df = pd.DataFrame(f1_data).sort_values('DistilBERT F1', ascending=True)

n = st.slider("Select number of Classes", 1,77,10)
fig, ax = plt.subplots(figsize=(10, 20))
df.tail(n).plot(kind='barh', ax=ax)
ax.set_xlabel('F1 Score')
ax.set_title('Per-Class F1 Score — TF-IDF vs DistilBERT')
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
