from bs4 import BeautifulSoup
import requests, re, random, os, csv, math, keras, warnings, datetime
from math import asin, atan2, cos, degrees, radians, sin
import plotly.io as pio
import plotly.express as px
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping
import base64
from streamlit_option_menu import option_menu
from datetime import datetime
import pytz

def get_download_link(file, x, type):
    b64 = base64.b64encode(file.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/{type};base64,{b64}" download = "{x}.{type}">Download {type.upper()} file of {x}</a>'
def create_dataset(dataset, look_back, look_ahead):
    X, Y = [], []
    for i in range(len(dataset)-look_back-look_ahead):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back + look_ahead - 1, 0])
    return np.array(X), np.array(Y)
def model_implementation(files, flight):
    dataframelist = []
    for i in files:
        df_new = pd.read_csv(r'Datasets/{}-{}.csv'.format(flight, i))
        #df_new = df_new.dropna(subset=['Time (IST)']).reset_index(drop=True)
        daylist = np.array(df_new['Time (EDT)'])
        strday = daylist[0][:3]
        df_new['date_time'] = np.nan
        for j in range(df_new.shape[0]):
            day2 = daylist[j][:3]
            if(strday==day2):
                df_new['date_time'][j] = i + daylist[j][3:]
            else:
                df_new['date_time'][j] = str(int(i[:2]) + 1) + i[2:] + daylist[j][3:]
        dataframelist.append(df_new)
    for df in dataframelist:
        df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M:%S')
        df['day'] = df['date_time'].apply(lambda x: x.day)
        df['hour'] = df['date_time'].apply(lambda x: x.hour)
        df['minute'] = df['date_time'].apply(lambda x: x.minute)
        df['second'] = df['date_time'].apply(lambda x: x.second)
    units = ['Latitude','Longitude','meters']
    imp_array = []
    predicted_df = dataframelist[-1]
    for i in units:
        arr = []
        pr = dataframelist[-1][i][:6]
        df_update = dataframelist[0].loc[:,['date_time',i, 'day', 'hour','minute','second', 'Course', 'tilt']]
        for df in dataframelist[1:]:
            df_lat=df.loc[:,['date_time',i, 'day', 'hour','minute','second', 'Course', 'tilt']]
            df_update = pd.concat([df_update, df_lat], axis=0)
        dataset = df_update[i].values #numpy.ndarray
        dataset = dataset.astype('float32')
        dataset = np.reshape(dataset, (-1, 1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        test_size = dataframelist[-1].shape[0]
        train_size = len(dataset) - test_size
        train, test = dataset[:train_size,:], dataset[train_size:,:]
        look_back = 5
        look_ahead = 1
        X_train, Y_train = create_dataset(train, look_back, look_ahead)
        X_test, Y_test = create_dataset(test, look_back, look_ahead)
        # reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        model = Sequential()
        model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(X_train, Y_train, epochs=16, batch_size=32, validation_data=(X_test, Y_test), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5)], verbose=1, shuffle=False)
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        # invert predictions
        train_predict = scaler.inverse_transform(train_predict)
        Y_train = scaler.inverse_transform([Y_train])
        test_predict = scaler.inverse_transform(test_predict)
        Y_test = scaler.inverse_transform([Y_test])
        predicted_df[i] = np.append(pr, test_predict[:,0])
        arr.append(f'Train Mean Absolute Error for {i}: {mean_absolute_error(Y_train[0], train_predict[:,0])}')
        arr.append(f'Train Root Mean Squared Error for {i}: {np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))}')
        arr.append(f'Test Mean Absolute Error for {i}: {mean_absolute_error(Y_test[0], test_predict[:,0])}')
        arr.append(f'Test Root Mean Squared Error for {i}: {np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))}')
        fig = plt.figure(figsize=(8,4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc='upper right')
        arr.append(fig)
        aa=[x for x in range(Y_test.shape[1])]
        fig1 = plt.figure(figsize=(8,4))
        plt.plot(aa, Y_test[0][:], marker='.', label="actual")
        plt.plot(aa, test_predict[:,0][:], 'r', label="prediction")
        # plt.tick_params(left=False, labelleft=True) #remove ticks
        plt.tight_layout()
        sns.despine(top=True)
        plt.subplots_adjust(left=0.07)
        plt.ylabel(i, size=15)
        plt.xlabel('Time step', size=15)
        plt.legend(fontsize=10)
        arr.append(fig1)
        imp_array.append(arr)
    predicted_df.to_csv(r"Datasets/{}-{}.csv".format(flight, 'Predicted'), index = False)
    return imp_array
	
def convertingToKML(file,s,e, flight):
    f1 = open(r"Datasets/{}-{}.csv".format(flight, file), 'r', encoding = 'utf-8')
    reader = csv.reader(f1)
    f2 = open(r"KML-Files/{}-{}.kml".format(flight, file),'w')
    f2.write(f"""<?xml version="1.0" encoding="UTF-8"?>
	    <kml xmlns="http://www.opengis.net/kml/2.2"
  	    xmlns:gx="http://www.google.com/kml/ext/2.2">
	    <name>{s} to {e}</name>    
		    <gx:Tour>
	        <name>{s} to {e}</name>
	   		    <gx:Playlist>\n""")
    x = 0
    for row in reader:
        if x == 0:
            x = 1
        
        else:
            f2.write(f"""				  <gx:FlyTo>
			    	        <gx:duration>{float(row[9])/20.0}</gx:duration>
				            <gx:flyToMode>smooth</gx:flyToMode>
			    	        <Camera>
			        	    <longitude>{row[2]}</longitude>
			        	    <latitude>{row[1]}</latitude>
			        	    <altitude>{row[6]}</altitude>
			        	    <heading>{row[3]}</heading>
			          	    <tilt>{90.00 + float(row[12])}</tilt>
						    <roll>0</roll>
			        	    <altitudeMode>absolute</altitudeMode>
			    	        </Camera>
			    	    </gx:FlyTo>\n""")
    f2.write("""		  </gx:Playlist>
	  	    </gx:Tour></kml>""")
    f1.close()
    f2.close()
    f1 = open(r"Datasets/{}-{}.csv".format(flight, file), 'r', encoding = 'utf-8')
    l1 = get_download_link(f1.read(), file, "csv")
    f1.close()
    f = open(r"KML-Files/{}-{}.kml".format(flight,file),'r')
    l2 = get_download_link(f.read(), file, "kml")
    f.close()
    return l1, l2
def time_difference(t1, t2):
    return (pd.to_datetime(t2) - pd.to_datetime(t1)).total_seconds()
def mph_to_mps(speed):
    return speed*4/9
def tilt_calculator(d, met1, met2):
    try:
        tilt = degrees(asin((met2-met1)/d))
    except Exception:
        tilt = 0
    return tilt
def distance_travelled_from_last_point(mps, time):
    return mps*time
# The formulas used here are from the following website: https://math.stackexchange.com/questions/463790/given-an-airplanes-latitude-longitude-altitude-course-dive-angle-and-speed?answertab=modifieddesc#tab-top
def get_point_at_distance(lat1, lon1, alt1, d, bearing, tilt, R=6371):
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    tilt = radians(tilt)
    
    alt2 = alt1 + d*1000*sin(tilt)
    lat2 = lat1 + (d/R)*cos(a)*cos(tilt)
    lon2 = lon1 + (d/(R*cos(lat1)))*sin(a)*cos(tilt)
    
    return (degrees(lat2), degrees(lon2), alt2, )
def scraping_function(url, s_elevation, e_elevation, flight, s, e):
    url_extract = requests.get(url).text
    soup = BeautifulSoup(url_extract, 'lxml')
    table = soup.find('table', class_ = "prettyTable fullWidth")
    header = table.find('tr', class_ = "thirdHeader")
    header_details = header.find_all('th')
    header_csv = []
    for column in header_details:
        try:
            texts = column.find('span', class_="show-for-medium-up").text
        except Exception:
            texts = column.text
        header_csv.append(texts)
    table_data = table.find_all('tr')
    table_data.pop(0)
    table_data.pop(0)
    table_data.pop(0)
    data_csv = []
    for row in table_data:
        columns = row.find_all('td')
        column_data = []
        for i in columns:
            try:
                tds = i.find('span').text
            except Exception:
                tds = i.text
            column_data.append(tds)
        if column_data[1] != '':
            column_data[3] = column_data[3].split(' ')[1][:-1]
            column_data[7] = column_data[7][:-1]
            if column_data[4] == '':
                column_data[4] = str(data_csv[-1][4])
                column_data[5] = str(data_csv[-1][5])
                column_data[6] = str(data_csv[-1][6])
            for i in range(1, len(column_data)-1):
                try:
                    column_data[i] = float(column_data[i].replace(',', ''))
                except Exception:
                    column_data[i] = 0
            data_csv.append(column_data)
    header_csv.extend(['Time Diff', 'm/s', 'Dist from lp', 'tilt'])
    data_csv[0].extend([0,mph_to_mps(data_csv[0][5]), 0,0])
    df = pd.DataFrame(columns = header_csv)
    for i in range(1, len(data_csv)):
        time = time_difference(data_csv[i-1][0][4:], data_csv[i][0][4:])
        mps = mph_to_mps(data_csv[i][5])
        d = distance_travelled_from_last_point(mps, time)
        tilt = tilt_calculator(d, data_csv[i-1][6], data_csv[i][6])
        data_csv[i].extend([time, mps, d, tilt])
    for i in range(len(data_csv)):
        df.loc[i] = data_csv[i]
    x = "{}-{}-{}".format(url[-27:-25], url[-29:-27], url[-33:-29])
    df.to_csv(r"Datasets/{}-{}.csv".format(flight, x), index = False)
    return x
    
    
def main_function(airport1, airport2):
    airports = pd.read_csv("in-airports.csv")
    main_url = "https://uk.flightaware.com"
    url_extract = requests.get(main_url + "/live/airport/{}".format(airport1)).text
    s = airports[airports['gps_code'] == airport1].reset_index(drop = True)['municipality'][0]
    e = airports[airports['iata_code'] == airport2].reset_index(drop = True)['municipality'][0]
    st.write(airport1, airport2, s, e)
    soup = BeautifulSoup(url_extract, 'lxml')
    tables = soup.find_all('div', class_ ="airportBoardContainer")[1::2]
    trs = []
    for i in tables:
        trs.extend(i.find_all('tr'))
    flights = []
    for i in trs:
        if airport2 in i.text:
            tds = i.find_all('td')
            flights.append(tds[0].text.replace(" ",""))
    st.write(flights)

    try:
        while True:
            flight = random.choice(flights)
            st.write(flight)
            url_extract = requests.get(main_url + "/live/flight/{}/history".format(flight)).text
            soup = BeautifulSoup(url_extract, 'lxml')
            new_table = soup.find('table', class_ = "prettyTable fullWidth tablesaw tablesaw-stack")
            table_body = new_table.find_all('tr')[1:]
            while True:
                if 'Scheduled' in table_body[0].text or 'On The Way!' in table_body[0].text:
                    table_body.pop(0)
                else:
                    break
            flight_links = []
            for i in table_body:
                if airport1 in i.text and airport2 in i.text and 'Cancelled' not in i.text:
                    x = re.findall(r'a href="[/a-zA-Z0-9]+', str(i))[0][8:]
                    flight_links.append(x)
                    if len(flight_links) == 5:
                        break
            elevation1 = airports[airports['gps_code'] == airport1].reset_index(drop=True)['elevation_ft'][0]*0.3048
            elevation2 = airports[airports['gps_code'] == airport2].reset_index(drop=True)['elevation_ft'][0]*0.3048
            if len(flight_links) == 5:
                fileslist = []
                for i in range(5):
                    file = scraping_function(main_url+flight_links[i]+"/tracklog", elevation1, elevation2, flight,s,e)
                    fileslist.insert(0, file)
                return fileslist, flight,s,e
            else:
                set1.remove(flight)
    except Exception as ex:
        st.write(f"No flights are there between {s} and {e}, change the locations and try again.")
        return
    
df = pd.read_csv("in-airports.csv")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.kukksi.de/wp-content/uploads/2021/02/iStock-807395598-1536x864.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
st.set_page_config(layout='wide', page_title="Bird's Eye", page_icon = "https://toppng.com/uploads/preview/transparent-background-airplane-11549404876oivb2vpwwf.png")
add_bg_from_url()
#st.write(datetime.now(pytz.timezone('Asia/Kolkata')))
tk = 0
st.title("Predict Flight Path Between Two Locations :airplane:")
col1, col2 = st.columns(2)
with col1:
    origin = st.selectbox('Origin: ', set(df['Display Name']), index = 0)
with col2:
    destination = st.selectbox('Destination: ', tuple(df[df['Display Name']!=origin]['Display Name']))
    if st.button('Submit'):
        tk = 1
if tk == 1:
    x = df[df['Display Name'] == origin].reset_index(drop=True)['gps_code'][0]
    y = df[df['Display Name'] == destination].reset_index(drop=True)['iata_code'][0]
    st.write(x)
    st.write(y)
    placeholder = st.empty()
    placeholder.text("Scraping is going on....")
    a_list, flight,s,e = main_function(x, y)
    if type(a_list) is list:
        placeholder.text("Scraping has been done successfully")
        st.write("Flight in consideration is {}".format(flight))
        placeholder.empty()
        placeholder = st.empty()
        placeholder.text("Model Training in progress....")
        x = model_implementation(a_list, flight)
        placeholder.text("Model Training successful")
        tab1, tab2 = st.tabs(["Prediction","Analysis"])
        placeholder.empty()
        tab1.markdown('<h1>Prediction:</h1>', unsafe_allow_html = True)
        l1, l2 = convertingToKML('Predicted', s, e, flight)
        tab1.markdown(l1, unsafe_allow_html = True)
        tab1.markdown(l2, unsafe_allow_html = True)
        for i in x:
            for j in i[:-2]:
                tab1.write(j)
            tab1.pyplot(i[-2])
            tab1.pyplot(i[-1])
        tab2.markdown('<h1>Analysis:</h1>', unsafe_allow_html = True)
        for i in a_list:
            df = pd.read_csv(r"Datasets/{}-{}.csv".format(flight, i))
            fig = px.line_3d(df, x="Longitude", y = "Latitude", z="meters", title = "Trajectory of the plane {} on {}".format(flight, i))
            tab2.plotly_chart(fig, use_container_width = True)
            l1, l2 = convertingToKML(i, s, e, flight)
            tab2.markdown(l1, unsafe_allow_html = True)
            tab2.markdown(l2, unsafe_allow_html = True)
