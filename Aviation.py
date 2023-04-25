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
import folium
from streamlit_folium import st_folium, folium_static
import streamlit.components.v1 as components

def get_download_link(file, x, type):
    b64 = base64.b64encode(file.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/{type};base64,{b64}" download = "{x}.{type}">Download {type.upper()} file of {x}</a>'
def create_dataset(dataset, look_back, look_ahead):
    X, Y = [], []
    for i in range(len(dataset)-look_back-look_ahead+1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back + look_ahead - 1, 0])
    return np.array(X), np.array(Y)

def df_creation(flight, file):
    df = pd.read_csv(r'Datasets/{}-{}.csv'.format(flight, file))
    #df_new = df_new.dropna(subset=['Time (IST)']).reset_index(drop=True)
    daylist = np.array(df['Time (EDT)'])
    strday = daylist[0][:3]
    df['date_time'] = np.nan
    for j in range(df.shape[0]):
        day2 = daylist[j][:3]
        if(strday==day2):
            df['date_time'][j] = file + daylist[j][3:]
        else:
            if(int(file[:2])!=31):
                df['date_time'][j] = str(int(file[:2]) + 1) + file[2:] + daylist[j][3:]
            else:
                df['date_time'][j] = "01-0" + str(int(file[4])+1) + file[5:] + daylist[j][3:]
    df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M:%S')
    df['day'] = df['date_time'].apply(lambda x: x.day)
    df['hour'] = df['date_time'].apply(lambda x: x.hour)
    df['minute'] = df['date_time'].apply(lambda x: x.minute)
    df['second'] = df['date_time'].apply(lambda x: x.second)
    return df

def model_implementation(files, flight, og, look_back, look_ahead):
    dataframelist = []
    for i in files:
        dataframelist.append(df_creation(flight, i))
    if og != None:
        og_df = df_creation(flight, og)
    units = ['Latitude','Longitude','Altitude(m)']
    units_dict = {}
    predicted_df = dataframelist[-1]
    og_p = []
    for i in units:
        arr = []
        pr = dataframelist[-1][i][:(look_back+look_ahead-1)]
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
        units_dict[i] = arr
        if og != None:
             og_lat=og_df.loc[:,['date_time',i, 'day', 'hour','minute','second', 'Course', 'tilt']]
             og_dataset =og_lat[i].values #numpy.ndarray
             og_dataset = og_dataset.astype('float32')
             og_dataset = np.reshape(og_dataset, (-1, 1))
             og_dataset = scaler.transform(og_dataset)
             og_test = []
             og_test.append(og_dataset[-look_back:, 0])
             og_test = np.array(og_test)
             og_test = np.reshape(og_test, (og_test.shape[0], 1, og_test.shape[1]))
             og_predict = model.predict(og_test)
             og_predict = scaler.inverse_transform(og_predict)
             og_p.append("The next {} is: {}".format(i, og_predict[0,0]))
    predicted_df.to_csv(r"Datasets/{}-{}.csv".format(flight, 'Predicted'), index = False)
    return units_dict, og_p
	
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
    for i in range(len(header_csv)):
        if header_csv[i] == 'meters':
            header_csv[i] = 'Altitude(m)'
    table_data = table.find_all('tr')
    table_data.pop(0)
    table_data.pop(0)
    table_data.pop(0)
    data_csv = []
    for row in table_data:
        try:
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
        except Exception:
            break
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
    soup = BeautifulSoup(url_extract, 'lxml')
    try:
        while True:
            flight = flights[0]
            ulo = main_url + "/live/flight/{}/history".format(flight)
            url_extract = requests.get(ulo).text
            soup = BeautifulSoup(url_extract, 'lxml')
            new_table = soup.find('table', class_ = "prettyTable fullWidth tablesaw tablesaw-stack")
            table_body = new_table.find_all('tr')[1:]
            while True:
                if 'Scheduled' in table_body[0].text:
                    table_body.pop(0)
                else:
                    break
            elevation1 = airports[airports['gps_code'] == airport1].reset_index(drop=True)['elevation_ft'][0]*0.3048
            elevation2 = airports[airports['iata_code'] == airport2].reset_index(drop=True)['elevation_ft'][0]*0.3048
            og = None
            if airport1 in table_body[0].text and airport2 in table_body[0].text and 'On The Way!' in table_body[0].text:
                x = re.findall(r'a href="[/a-zA-Z0-9]+', str(table_body[0]))[0][8:]
                og = scraping_function(main_url+x+"/tracklog",elevation1,elevation2,flight,s,e)
            table_body.pop(0)
            flight_links = []
            for i in table_body:
                if airport1 in i.text and airport2 in i.text and 'Cancelled' not in i.text:
                    x = re.findall(r'a href="[/a-zA-Z0-9]+', str(i))[0][8:]
                    flight_links.append(x)
                    if len(flight_links) == 5:
                        break
            if len(flight_links) == 5:
                fileslist = []
                for i in range(5):
                    file = scraping_function(main_url+flight_links[i]+"/tracklog", elevation1, elevation2, flight,s,e)
                    fileslist.insert(0, file)
                return fileslist,flight,s,e,og
            else:
                flights.remove(flight)
    except Exception as ex:
        st.write(f"No flights are there between {s} and {e}, change the locations and try again.")
        return "", "", "", "", ""
def destination_maker(origin):
    airports = pd.read_csv("in-airports.csv")
    main_url = "https://uk.flightaware.com"
    st.write(origin)
    url_extract = requests.get(main_url + "/live/airport/{}".format(origin)).text
    soup = BeautifulSoup(url_extract, 'lxml')
    tables = soup.find_all('div', class_ ="airportBoardContainer")[1::2]
    trs = []
    for i in tables:
        trs.extend(i.find_all('tr'))
    trs.pop(0)
    trs.pop(0)
    flights = pd.DataFrame(columns = ['iata_code', 'Display Name', 'Flight'])
    for i in trs:
        tds = i.find_all('td')
        st.write(tds)
        st.write(tds[2])
        tds[2] = re.findall(r"\(.\)", tds[2].text)
        st.write(tds[2])
        if tds[2] in airports['iata_code']:
            flights.loc[flights.shape[0]] = [tds[2],airports[airports['iata_code'] == tds[2]].reset_index(drop=True)['Display Name'][0] ,tds[0].text.replace(" ","")]
    return flights
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

def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('p'), i;
                for (i = 0; i < elements.length; ++i) 
                    { if (elements[i].textContent.includes(|wgt_txt|)) 
                        { elements[i].style.fontSize ='""" + wch_font_size + """'; } }</script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

st.set_page_config(layout='wide', page_title="Bird's Eye", page_icon = "https://toppng.com/uploads/preview/transparent-background-airplane-11549404876oivb2vpwwf.png")
add_bg_from_url()
#st.write(datetime.now(pytz.timezone('Asia/Kolkata')))
tk = 0
st.title("Predict Flight Path Between Two Locations :airplane:")
col1, col2, col3 = st.columns(3)
with col1:
    origin = st.selectbox('Origin: ', set(df['Display Name']), index = 0)
    x = df[df['Display Name'] == origin].reset_index(drop=True)['gps_code'][0]
    look_behind = st.slider('Look Behind',5,20, help = "It is the past n data points being used for making the next prediction.")
with col2:
    dest = destination_maker(x)
    destination = st.selectbox('Destination: ', tuple(dest['Display Name']))
    y = dest[dest['Display Name'] == destination].reset_index(drop=True)['iata_code'][0]
    look_ahead = st.slider("Look Ahead",1,10, help = "It is the nth future point being predicted, in intervals of 30 seconds.")
with col3:
    flight = st.selectbox('Flight: ', tuple(dest[dest['Display Name'] == destination]['Flight']))
    if st.button('Submit'):
        tk = 1
if tk == 1:
    x = df[df['Display Name'] == origin].reset_index(drop=True)['gps_code'][0]
    y = df[df['Display Name'] == destination].reset_index(drop=True)['iata_code'][0]
    placeholder = st.empty()
    placeholder.text("Scraping is going on....")
    a_list, flight,s,e,og = main_function(x, y)
    if type(a_list) is list:
        placeholder.text("Scraping has been done successfully")
        st.write("Flight in consideration is {}".format(flight))
        placeholder.empty()
        placeholder = st.empty()
        placeholder.text("Model Training in progress....")
        results, og_p = model_implementation(a_list, flight, og, look_behind, look_ahead)
        placeholder.text("Model Training successful")
        listTabs = ['Prediction','Ongoing Flight','Analysis']
        tab1, tab2, tab3 = st.tabs(listTabs)
        ChangeWidgetFontSize(listTabs[0], '24px')
        ChangeWidgetFontSize(listTabs[1], '24px')
        ChangeWidgetFontSize(listTabs[2], '24px')
        placeholder.empty()
        with tab1:
            df = pd.read_csv(r"Datasets/{}-Predicted.csv".format(flight))
            st.markdown('<h1>Prediction:</h1>', unsafe_allow_html = True)
            st.subheader("Predicted Flight")
            st.markdown("<h4>Trajectory:</h4>", unsafe_allow_html = True)
            fig = px.line_3d(df, x="Longitude", y = "Latitude", z="Altitude(m)")
            st.plotly_chart(fig)
            st.markdown("<h4>Path:</h4>", unsafe_allow_html = True)
            m = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()],zoom_start=5,control_scale=True)
            loc = []
            for r,rows in df.iterrows():
                loc.append((rows['Latitude'], rows['Longitude']))
            folium.PolyLine(loc, color = 'red', weight=5, opacity = 0.8).add_to(m)
            folium_static(m)
            x = "Download Files:"
            exp = st.expander(x)
            ChangeWidgetFontSize(x, '20px')
            with exp:
                l1, l2 = convertingToKML('Predicted', s, e, flight)
                st.markdown(l1, unsafe_allow_html = True)
                st.markdown(l2, unsafe_allow_html = True)
            for i in results.keys():
                st.subheader("Model for {}:".format(i))
                for j in results[i][:-2]:
                    st.write(j)
                    ChangeWidgetFontSize(j, '20px')
                st.pyplot(results[i][-2])
                st.pyplot(results[i][-1])
        with tab2:
            st.markdown("<h1>Ongoing Flight:</h1>", unsafe_allow_html=True)
            if og != None:
                df = pd.read_csv(r"Datasets/{}-{}.csv".format(flight,og))
                for p in og_p:
                    tab2.write(p)
                st.markdown("<h4>Trajectory:</h4>", unsafe_allow_html = True)
                fig = px.line_3d(df, x="Longitude", y = "Latitude", z="Altitude(m)")
                st.plotly_chart(fig)
                st.markdown("<h4>Path:</h4>", unsafe_allow_html = True)
                m = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()],zoom_start=5,control_scale=True)
                loc = []
                for r,rows in df.iterrows():
                    loc.append((rows['Latitude'], rows['Longitude']))
                folium.PolyLine(loc, color = 'red', weight=5, opacity = 0.8).add_to(m)
                folium_static(m)
            else:
               st.subheader("This flight is not ongoing, it is scheduled to start soon.")
        with tab3:
            st.markdown('<h1>Analysis:</h1>', unsafe_allow_html = True)
            for i in a_list:
                df = pd.read_csv(r"Datasets/{}-{}.csv".format(flight, i))
                st.subheader("Flight on {}".format(i))
                st.markdown("<h4>Trajectory:</h4>", unsafe_allow_html = True)
                fig = px.line_3d(df, x="Longitude", y = "Latitude", z="Altitude(m)")
                st.plotly_chart(fig)
                st.markdown("<h4>Path:</h4>", unsafe_allow_html = True)
                m = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()],zoom_start=5,control_scale=True)
                loc = []
                for r,rows in df.iterrows():
                    loc.append((rows['Latitude'], rows['Longitude']))
                folium.PolyLine(loc, color = 'red', weight=5, opacity = 0.8).add_to(m)
                folium_static(m)
                x = "Download Files:"
                exp = st.expander(x)
                ChangeWidgetFontSize(x, '20px')
                with exp:
                    l1, l2 = convertingToKML(i, s, e, flight)
                    st.markdown(l1, unsafe_allow_html = True)
                    st.markdown(l2, unsafe_allow_html = True)
