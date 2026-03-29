import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Amazon Lugage Review",page_icon="🛍️", layout="wide")
st.title("💼 Amazon Luggage Market Intelligence")
st.markdown("AI-powered insights derived from real customer reviews and pricing data.")

@st.cache_data
def read_data():
    brand_summary=pd.read_csv("brand_summary.csv")
    product_summary=pd.read_csv("product_sentiment.csv")
    agent_insight=pd.read_csv("agent_insights.csv")
    return brand_summary,product_summary,agent_insight

brand,product,agent=read_data()
st.sidebar.image("i3.png",width=200)
st.sidebar.title("Settings⚙️")

all_brands=brand['brand'].unique().tolist()
selected_brands=st.sidebar.multiselect("Select the barnds to comapre",options=all_brands,default=all_brands)
filtered_brands=brand[brand["brand"].isin(selected_brands)]
filtered_products=product[product["brand"].isin(selected_brands)]
st.sidebar.divider()
user_key=st.sidebar.text_input("Enter Google Api Key",type="password")
st.sidebar.caption("Get your free Gemini API key [here](https://aistudio.google.com/app/apikey) to enable the Ai Assistant.")

tab1,tab2,tab3,tab4=st.tabs(["📊 Executive Summary", "🥊 Comparison", "🔍 Product Deep Dive","🤖 AI Assistant"])

with tab1:
    st.header("Market Overview")
    total_brands_selected=filtered_brands["brand"].nunique()
    total_products=filtered_brands['total_products'].sum()
    total_reviews_analyzed = filtered_brands['total_reviews'].sum()

    if total_brands_selected>0:
        avg_market_sentiment = filtered_brands['avg_sentiment_score'].mean()
    else:
        avg_market_sentiment=0
    col1,col2,col3,col4=st.columns(4)
    col1.metric("Brands Tracked", total_brands_selected)
    col2.metric("Total Products", total_products)
    col3.metric("Verified Reviews", f"{total_reviews_analyzed:,}")
    col4.metric("Avg Market Sentiment", f"{avg_market_sentiment:.1f} / 10")

    st.divider()

    st.subheader("🤖 Agent Insight")
    if selected_brands:
        search_pattern = '|'.join(selected_brands)
        mask = agent['title'].str.contains(search_pattern, case=False, na=False) | \
               agent['detail'].str.contains(search_pattern, case=False, na=False)
        filtered_insights = agent[mask]
    else:
        filtered_insights = pd.DataFrame(columns=agent.columns)
    if not filtered_insights.empty:
        for index,row in filtered_insights.iterrows():
            with st.expander(f"💡 Insight {row['insight_number']}: {row['title']}", expanded=True):
                st.markdown(f"**The Data Story:** {row['detail']}")
                st.success(f"**Strategic Implication:** {row['implication']}")
    else:
        st.info("Select a brand from the sidebar to view specific strategic insights.")

with tab2:
    st.header("Comparison")
    st.markdown("Analyze how brands position themselves in terms of pricing, sentiment, and discounting strategies.")

    if not filtered_brands.empty:
        
        st.subheader("Value vs. Sentiment Matrix")
        st.info("💡 **How to read this:** Brands in the **top-left** offer high satisfaction at a lower price (Value Winners). Brands in the **top-right** are successful premium players. Bubbles are sized by the total number of reviews.")

        fig_scatter = px.scatter(
            filtered_brands,
            x="avg_selling_price",
            y="avg_sentiment_score",
            color="brand",
            size="total_reviews",
            hover_name="brand",
            text="brand",
            labels={
                "avg_selling_price": "Average Selling Price (₹)",
                "avg_sentiment_score": "AI Sentiment Score (1-10)"
            },
            height=500
        )
        fig_scatter.update_traces(textposition='top center')
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("The 'Discount Trap' Analysis")
            st.markdown("Which brand relies on the heaviest discounts to sell?")
            
            fig_bar = px.bar(
                filtered_brands,
                x="brand",
                y="avg_discount_pct",
                color="brand",
                text_auto='.1f',
                labels={
                    "brand": "Brand",
                    "avg_discount_pct": "Average Discount (%)"
                }
            )
            fig_bar.update_layout(showlegend=False) 
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Top Extracted Feedback")
            st.markdown("The most common themes extracted by the LLM.")
            
            feedback_df = filtered_brands[['brand', 'top_positives', 'top_negatives']].copy()
            feedback_df.columns = ['Brand', 'Top Pros', 'Top Cons']
            
            st.dataframe(feedback_df, hide_index=True, use_container_width=True)

    else:
        st.warning("Please select at least one brand from the sidebar to view competitive metrics.")


with tab3:
    st.header("Product Deep Dive")
    st.markdown("Select a specific product to see its individual pricing, performance, and detailed AI review summary.")

    if not filtered_products.empty:
        product_list = filtered_products['title'].unique().tolist()
        selected_product_title = st.selectbox("Search for a Product:", product_list)
        
        product_data = filtered_products[filtered_products['title'] == selected_product_title].iloc[0]
        
        st.subheader(f"{product_data['brand']}")
        st.markdown(f"**{selected_product_title}**")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selling Price", f"₹{product_data['selling_price']}")
        col2.metric("Discount", f"{product_data['discount_pct']}%")
        col3.metric("Star Rating", f"{product_data['actual_rating']} ⭐")
        col4.metric("AI Sentiment", f"{product_data['sentiment_score']} / 10")
        
        st.divider()
        
        st.info(f"**🤖 AI One-Line Verdict:** {product_data['one_line_verdict']}")
        
        st.write("")
        
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.success("✅ **Top Praises**")
            st.write(product_data['top_positives']) 
            
        with col_cons:
            st.error("⚠️ **Top Complaints**")
            st.write(product_data['top_negatives'])
            
        st.divider()
        
        st.subheader("Component Quality Scores (Out of 10)")
        feature_cols = st.columns(4)
        feature_cols[0].metric("🛞 Wheels", product_data.get('asp_wheels', 'N/A'))
        feature_cols[1].metric("🧳 Handle", product_data.get('asp_handle', 'N/A'))
        feature_cols[2].metric("🤐 Zipper", product_data.get('asp_zipper', 'N/A'))
        feature_cols[3].metric("🛡️ Durability", product_data.get('asp_durability', 'N/A'))
        
    else:
        st.warning("Please select at least one brand from the sidebar to view specific products.")


with tab4:
    st.header("AI Assistant")
    st.info("Note: This is not a conversational agent so it doesn't answer followup questions")
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message["content"])
    if user_key:
        if prompt:=st.chat_input("Enter your query"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=user_key, temperature=0.2)
                    request=f'''
                        DATA PROVIDED:BRAND Data{brand}, PRODUCT DATA:{product}, UNSER PROMPT{prompt}
                        ROLE:YPU ARE A SENIOR REVIEW AGENT WHO CAN UNDERSTAND THE PRODUCT SENTIMENT AND ANSWER THE USER QUERY
                        TASK: UNDERSTAND THE USER PROMPT GO THROUGH THE PROVIDED DATA AND ANSWER ONLY FROM THE PROVIDED DATA REGARDING THE PRODUCT OR BRAND USER ASKS FOR
                        IF USER ASKS FOR THE BRAND OUTSIDE THE DATA POLITELY IGNORE TO ANSWER ANS ASK TO SELECT FROM OF THESE BRANDS{all_brands}
                        RETURN A COMPRAHENSIVE DRAFTED USEFUL RESPONSE 
                        '''
                    response=llm.invoke(request)
                    st.markdown(response.content)
                st.session_state.messages.append({"role":"assistant","content":response.content})
    else:
        st.warning("Please Enter Your Google Api key")