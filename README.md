# OpenTelemetry Demo
This project is an example of how to set up open telemetry with your Python application. 
    ![](./screenshots/dfd.png)
## Prerequisites
- Docker
- Docker Compose
- Python3

## Installation
Make sure you have Docker installed. Then, clone the repo and run the following command.

    docker compose up --build -d

This will build and start the following services:

- Grafana Tempo: Distributed tracing backend
- Prometheus: Metrics backend
- Grafana: Visualization frontend
- Python Example App

## Functionality
This Python app allows you to demo how open telemetry works. It consists of several endpoints that demonstrate different functionalities and creates a unique trace for each bit. Here is a brief description of each of the functionalities:

- **Tracing**: Tracks the requests and shows a detailed report of each step.
- **Filter**: Allows you to filter the traces based on different parameters.

### Functional requirement

|   | 功能                                                          | 
|---|-------------------------------------------------------------| 
| 1 | 以D3.js 來顯示知識圖關鍵字之間的關連         |  
| 2 |  使用opentelemetry, tempo, prometheus, grafana 記錄數效能及事件logging  |  
| 3 | Filter: 選取及顯示特定關鍵字                                          |  
| 4 | Filter: 調整顯示的關鍵字數量                                          |  
| 5 | Filter: 調整顯示的關鍵字來源年份                                        |  
| 6 | Filter: 調整顯示的關鍵字最低的共現數                                      |  
| 7 | 以Neo4j 圖像DB 來存關鍵字資料                                         |  

### Non-Functional requirement

|  |  功能 | Supporting Description                               |
| --- | --- |------------------------------------------------------|
| 1 | load-balancing | 三個 python container: nginx, web-server, db-service |

### trace
|    | trace | Supporting Description |
|----| --- | --- |

| 1  | is_production_environment | 是不是在production 環境 |
| 2  | selected_items | Filter: 選取及顯示特定關鍵字 |
| 3  | limit | Filter: 調整顯示的關鍵字數量 |
| 4  | skip | Filter: 為關鍵字的分頁顯示功能 |
| 5  | years | Filter: 調整顯示的關鍵字來源年份 |
| 6  | co-occurrence | Filter: 調整顯示的關鍵字最低的共現數 |
| 7  | condition | 資料庫的select 條件 |
| 8  | api_server_url | back-end server 的IP位置 |
| 9  | file_path | configure 的位置 |
| 10 | cypher_query | Neo4j Database query |
| 11 | result_count | 資料庫search result 的筆數 |
| 12 | datarun_id | project 的唯一區別ID |
| 13 | number_of_clusters | 關鍵字分群的群組數量 |
| 14 | transformed_results_count | 轉換為D3.js 格式後的資料數量 |



## Usage
After setting up the services, you can generate traces by making requests to the app:

     http://127.0.0.1:5000/ 

To view the traces:

1. Open Grafana at http://localhost:3000/
2. Open the Explore page and select Tempo as the data source
3. Click the Run query button to see the traces