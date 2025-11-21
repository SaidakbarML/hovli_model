import streamlit as st
import pandas as pd
import pickle
from targetenc import TargetEncoding




@st.cache_resource
def load_target_encoder():
    with open("target_encoder.pkl", "rb") as f:
        enc = pickle.load(f)
    return enc


@st.cache_resource
def load_model():
    with open("lgbm_model_v2.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_target_encoder():
    with open("target_encoder.pkl", "rb") as f:
        enc = pickle.load(f)
    return enc

model = load_model()
target_encoder = load_target_encoder()

model_features = list(model.feature_names_in_)

st.title("🏠 House Price Prediction App")



#  INPUT FORM

st.subheader("Enter property details:")

business = st.selectbox("Is the property business-related?", [0, 1])
price_negotiable = st.selectbox("Price negotiable?", [0, 1])
number_of_rooms = st.number_input("Number of rooms", 1, 20)
total_area = st.number_input("Total area (m²)", 1, 50000)
total_living_area = st.number_input("Living area (m²)", 1, 10000)
total_floors = st.number_input("Total floors", 1, 30)
ceiling_height = st.number_input("Ceiling height (cm)", 200, 400)
furnished_house = st.selectbox("Furnished?", [0, 1])
year_of_construction_sale = st.number_input("Year built", 1900, 2025)
comission = st.selectbox("Commission included?", [0, 1])

# MULTISELECTS
house_repairs = st.multiselect(
    "House Repairs",
    [
        "Авторский проект",
        "Евроремонт",
        "Не достроен",
        "Под снос",
        "Предчистовая отделка",
        "Средний ремонт",
        "Требует ремонта",
        "Черновая отделка"
    ]
)

more_house = st.multiselect(
    "Additional Features",
    [
        "AirConditioner",
        "Basement",
        "Bathhouse",
        "Garage",
        "Garden",
        "Gym",
        "Home appliances",
        "Internet",
        "Satellite TV",
        "Sauna",
        "Security",
        "Sewerage",
        "Storage room",
        "Swimming pool",
        "Telephone"
    ]
)

near_is = st.multiselect(
    "Nearby",
    [
        "Bus stops",
        "Cafe",
        "Entertainment venues",
        "Green area",
        "Hospital",
        "Kindergarten",
        "Park",
        "Parking lot",
        "Playground",
        "Polyclinic",
        "Restaurants",
        "School",
        "Shops",
        "Supermarket"
    ]
)

location_district_name = st.text_input("District name")



#  ENCODING FUNCTIONS

def encode_multilabel(df, column_name, classes_dict):
    for cls in classes_dict:
        df[f"{column_name}_{cls}"] = df[column_name].apply(lambda x: 1 if cls in x else 0)
    df.drop(columns=[column_name], inplace=True)
    return df


def build_input_df():

    df = pd.DataFrame([{
        "business": business,
        "price_negotiable": price_negotiable,
        "number_of_rooms": number_of_rooms,
        "total_area": total_area,
        "total_living_area": total_living_area,
        "total_floors": total_floors,
        "ceiling_height": ceiling_height,
        "furnished_house": furnished_house,
        "year_of_construction_sale": year_of_construction_sale,
        "comission": comission,
        "house_repairs": house_repairs,
        "more_house": more_house,
        "near_is": near_is,
        "location_district_name": location_district_name
    }])

    repair_classes = [
        "Авторский проект", "Евроремонт", "Не достроен", "Под снос",
        "Предчистовая отделка", "Средний ремонт", "Требует ремонта",
        "Черновая отделка"
    ]

    more_house_classes = [
        "Air_conditioner", "Basement", "Bathhouse", "Garage", "Garden", "Gym",
        "Home_appliances", "Internet", "Satellite_TV", "Sauna", "Security",
        "Sewerage", "Storage_room", "Swimming_pool", "Telephone"
    ]

    near_is_classes = [
        "Bus_stops", "Cafe", "Entertainment_venues", "Green_area", "Hospital",
        "Kindergarten", "Park", "Parking_lot", "Playground", "Polyclinic",
        "Restaurants", "School", "Shops", "Supermarket"
    ]

    # Encode MultiLabel
    df["house_repairs"] = df["house_repairs"].apply(lambda x: x if isinstance(x, list) else [])
    df = encode_multilabel(df, "house_repairs", repair_classes)

    df["more_house"] = df["more_house"].apply(lambda x: x if isinstance(x, list) else [])
    df = encode_multilabel(df, "more_house", more_house_classes)

    df["near_is"] = df["near_is"].apply(lambda x: x if isinstance(x, list) else [])
    df = encode_multilabel(df, "near_is", near_is_classes)

    # TARGET ENCODING (1 column)
    encoded = target_encoder.transform(
        df[["location_district_name"]],
        ["location_district_name"]
    )

    df["location_district_name_tencoded"] = encoded["location_district_name_tencoded"]
    df = df.drop(columns=["location_district_name"])

    return df


#  PREDICT BUTTON

if st.button("Predict Price"):

    df = build_input_df()

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns exactly as model expects
    df = df[model_features]

    pred = model.predict(df)[0]

    st.success(f"💰 Predicted Price: ${pred:,.0f} (+- error%)")

