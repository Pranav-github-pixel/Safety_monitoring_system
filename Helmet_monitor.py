import requests
import json
import time
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Configuration variables
DB_NAME = "helmet_data.db"
UPDATE_INTERVAL = 2000  # milliseconds (2 secs)


# Merge Sort implementation for data processing
def merge_sort(arr):
    """
    Sort an array using the Merge Sort algorithm.

    Args:
        arr: List of values to sort

    Returns:
        Sorted list
    """
    if len(arr) <= 1:
        return arr

    # Divide the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Recursively sort both halves
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)

    # Merge the sorted halves
    return merge(left_half, right_half)


def merge(left, right):
    """
    Merge two sorted arrays into a single sorted array.

    Args:
        left: First sorted array
        right: Second sorted array

    Returns:
        Merged sorted array
    """
    result = []
    i = j = 0

    # Compare elements from both arrays and add the smaller one to result
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])

    return result


# Longest Common Subsequence implementation for pattern detection
def lcs(sequence1, sequence2, threshold=0.1):
    """
    Find the Longest Common Subsequence between two numerical sequences.

    Args:
        sequence1: First numerical sequence (list of float values)
        sequence2: Second numerical sequence (list of float values)
        threshold: Tolerance for considering two values as equal (default: 0.1)

    Returns:
        Length of LCS and the actual subsequence
    """
    m = len(sequence1)
    n = len(sequence2)

    # Initialize the LCS matrix
    L = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Build the LCS matrix in bottom-up fashion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Consider values equal if they are within threshold
            if abs(sequence1[i - 1] - sequence2[j - 1]) <= threshold:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Reconstruct the LCS
    lcs_sequence = []
    i, j = m, n

    while i > 0 and j > 0:
        # If current characters are same
        if abs(sequence1[i - 1] - sequence2[j - 1]) <= threshold:
            lcs_sequence.append(sequence1[i - 1])
            i -= 1
            j -= 1
        # If not same, move in direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # Reverse the sequence to get it in correct order
    lcs_sequence.reverse()

    return L[m][n], lcs_sequence


# Application for pattern analysis
def analyze_jerk_patterns(jerk_data, reference_patterns):
    """
    Analyze jerk data for known patterns using LCS.

    Args:
        jerk_data: Recent jerk values from sensors
        reference_patterns: Dictionary of named reference patterns

    Returns:
        Dictionary with pattern names and match percentages
    """
    results = {}

    for pattern_name, pattern_data in reference_patterns.items():
        lcs_length, lcs_seq = lcs(jerk_data, pattern_data)
        match_percentage = (lcs_length / len(pattern_data)) * 100
        results[pattern_name] = {
            'match_percentage': match_percentage,
            'common_sequence': lcs_seq
        }

    return results


def detect_anomalies_with_lcs(current_data, historical_data, threshold=0.1):
    """
    Detect anomalies by comparing current data with historical patterns.

    Args:
        current_data: Recent sensor readings
        historical_data: Previous normal sensor readings
        threshold: Similarity threshold

    Returns:
        Anomaly score (lower LCS length indicates more anomalous behavior)
    """
    lcs_length, _ = lcs(current_data, historical_data, threshold)
    max_possible_length = min(len(current_data), len(historical_data))
    anomaly_score = 1 - (lcs_length / max_possible_length)

    return anomaly_score


class HelmetMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Safety Helmet Monitoring System")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ESP32 connection variables
        self.esp32_ip = tk.StringVar()
        self.connected = False
        self.data_collection_active = False
        self.collection_thread = None

        # Add reference patterns for LCS analysis
        self.reference_patterns = {
            'normal_movement': [0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.1],
            'fall_pattern': [0.2, 0.5, 1.2, 2.5, 3.0, 2.8, 1.5],
            'impact_pattern': [0.1, 0.2, 3.5, 3.2, 1.5, 0.5, 0.2]
        }

        # Create database
        self.setup_database()

        # Create UI
        self.create_ui()

        # Setup plots
        self.setup_plots()

        # Start animation
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=UPDATE_INTERVAL)

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Connection frame
        conn_frame = ttk.LabelFrame(main_frame, text="ESP32 Connection", padding="10")
        conn_frame.pack(fill=tk.X, pady=5)

        ttk.Label(conn_frame, text="ESP32 IP:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.esp32_ip, width=15).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.conn_button = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.conn_button.grid(row=0, column=2, padx=5, pady=5)

        self.status_label = ttk.Label(conn_frame, text="Status: Disconnected", foreground="red")
        self.status_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Current readings frame
        readings_frame = ttk.LabelFrame(main_frame, text="Current Readings", padding="10")
        readings_frame.pack(fill=tk.X, pady=5)

        # Temperature
        ttk.Label(readings_frame, text="Temperature:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.temp_label = ttk.Label(readings_frame, text="-- °C")
        self.temp_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Pressure
        ttk.Label(readings_frame, text="Pressure:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.pressure_label = ttk.Label(readings_frame, text="-- hPa")
        self.pressure_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Jerk
        ttk.Label(readings_frame, text="Jerk:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.jerk_label = ttk.Label(readings_frame, text="-- m/s³")
        self.jerk_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Jerk Status
        ttk.Label(readings_frame, text="Jerk Status:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.jerk_status_label = ttk.Label(readings_frame, text="Normal", foreground="blue")
        self.jerk_status_label.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # GPS
        ttk.Label(readings_frame, text="GPS:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.gps_label = ttk.Label(readings_frame, text="Lat: --, Lng: --")
        self.gps_label.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Signal Strength
        ttk.Label(readings_frame, text="Signal Strength:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.rssi_label = ttk.Label(readings_frame, text="-- dBm")
        self.rssi_label.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Last Update
        ttk.Label(readings_frame, text="Last Update:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.timestamp_label = ttk.Label(readings_frame, text="--")
        self.timestamp_label.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        # Data management frame
        data_frame = ttk.LabelFrame(main_frame, text="Data Management", padding="10")
        data_frame.pack(fill=tk.X, pady=5)

        self.export_button = ttk.Button(data_frame, text="Export to CSV", command=self.export_to_csv)
        self.export_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_button = ttk.Button(data_frame, text="Clear Database", command=self.clear_database)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Algorithm analysis frame
        algo_frame = ttk.LabelFrame(main_frame, text="Algorithm Analysis", padding="10")
        algo_frame.pack(fill=tk.X, pady=5)

        # LCS Analysis Button
        self.lcs_button = ttk.Button(algo_frame, text="Run Movement Analysis", command=self.run_lcs_analysis)
        self.lcs_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Merge Sort Analysis Button
        self.sort_button = ttk.Button(algo_frame, text="Database Analysis", command=self.run_merge_sort_analysis)
        self.sort_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Performance Analysis Button
        self.perf_button = ttk.Button(algo_frame, text="Algorithm Performance Analysis",
                                      command=self.analyze_algorithm_performance)
        self.perf_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Results Text Area
        self.results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        self.results_frame.pack(fill=tk.X, pady=5)

        self.results_text = tk.Text(self.results_frame, height=5, width=80)
        self.results_text.pack(fill=tk.X)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Canvas for matplotlib plots
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    def setup_plots(self):
        # Create matplotlib figure and embed in tkinter
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)

        # Setup plot areas
        self.ax1.set_ylabel('Temperature (°C)')
        self.ax1.set_title('Temperature')
        self.ax1.grid(True)

        self.ax2.set_ylabel('Pressure (hPa)')
        self.ax2.set_title('Pressure')
        self.ax2.grid(True)

        self.ax3.set_ylabel('Jerk (m/s³)')
        self.ax3.set_title('Jerk Detection')
        self.ax3.grid(True)

        # Add a red line for jerk threshold
        self.ax3.axhline(y=1.5, color='r', linestyle='--', alpha=0.7)
        self.ax3.text(0.02, 1.6, 'Threshold', color='r')

        # Embed plot in tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        toolbar.update()

    def setup_database(self):
        """Create SQLite database and tables if they don't exist"""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Create table for sensor data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            temperature REAL,
            pressure REAL,
            jerk_value REAL,
            latitude REAL,
            longitude REAL,
            rssi INTEGER
        )
        ''')

        conn.commit()
        conn.close()
        print("Database setup complete")

    def toggle_connection(self):
        """Connect to or disconnect from ESP32"""
        if not self.connected:
            # Try to connect
            ip = self.esp32_ip.get()
            if self.test_connection(ip):
                self.connected = True
                self.conn_button.config(text="Disconnect")
                self.status_label.config(text="Status: Connected", foreground="green")

                # Start data collection thread
                self.data_collection_active = True
                self.collection_thread = threading.Thread(target=self.collect_data, daemon=True)
                self.collection_thread.start()
            else:
                messagebox.showerror("Connection Error", f"Could not connect to ESP32 at {ip}")
        else:
            # Disconnect
            self.connected = False
            self.data_collection_active = False
            self.conn_button.config(text="Connect")
            self.status_label.config(text="Status: Disconnected", foreground="red")

    def test_connection(self, ip):
        """Test connection to ESP32"""
        try:
            # Try to connect to the data endpoint
            response = requests.get(f"http://{ip}/data", timeout=5)
            return response.status_code == 200
        except:
            return False

    def collect_data(self):
        """Thread function to collect data from ESP32"""
        while self.data_collection_active:
            try:
                data = self.fetch_data()
                if data:
                    self.save_to_database(data)
                    self.update_current_readings(data)
            except Exception as e:
                print(f"Error collecting data: {e}")

            time.sleep(2)  # Match ESP32's update interval

    def fetch_data(self):
        """Fetch data from ESP32 server"""
        try:
            ip = self.esp32_ip.get()
            response = requests.get(f"http://{ip}/data", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: Received status code {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return None

    def save_to_database(self, data):
        """Save data to SQLite database"""
        if not data:
            return False

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO sensor_data (timestamp, temperature, pressure, jerk_value, latitude, longitude, rssi)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            data.get('temperature', 0),
            data.get('pressure', 0),
            data.get('jerkValue', 0),
            data.get('latitude', 0),
            data.get('longitude', 0),
            data.get('rssi', 0)
        ))

        conn.commit()
        conn.close()
        return True

    def update_current_readings(self, data):
        """Update the current readings display"""
        # Use tkinter's after method to update UI from a non-main thread
        self.root.after(0, lambda: self._update_labels(data))

    def _update_labels(self, data):
        """Update UI labels with new data (called from main thread)"""
        self.temp_label.config(text=f"{data.get('temperature', 0):.1f} °C")
        self.pressure_label.config(text=f"{data.get('pressure', 0):.1f} hPa")
        self.jerk_label.config(text=f"{data.get('jerkValue', 0):.2f} m/s³")

        # Update jerk status
        if data.get('jerkValue', 0) > 0:
            self.jerk_status_label.config(text="JERK DETECTED!", foreground="red")
        else:
            self.jerk_status_label.config(text="Normal", foreground="blue")

        # Update GPS
        self.gps_label.config(text=f"Lat: {data.get('latitude', 0):.6f}, Lng: {data.get('longitude', 0):.6f}")

        # Update RSSI
        self.rssi_label.config(text=f"{data.get('rssi', 0)} dBm")

        # Update timestamp
        self.timestamp_label.config(text=data.get('timestamp', '--'))

    def get_recent_data(self, limit=100):
        """Retrieve recent data from database"""
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(f"SELECT * FROM sensor_data ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        # Reverse to get chronological order
        return df.iloc[::-1]

    def update_plots(self, frame):
        """Update the matplotlib plots with new data"""
        # Get the latest data from database
        df = self.get_recent_data(50)  # Last 50 records

        if df.empty:
            return self.ax1, self.ax2, self.ax3

        # Convert timestamps to datetime objects for better x-axis formatting
        df['datetime'] = pd.to_datetime(df['timestamp'])

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Plot the data
        self.ax1.plot(df['datetime'], df['temperature'], 'b-')
        self.ax2.plot(df['datetime'], df['pressure'], 'g-')

        # For jerk, use bar chart with color coding
        jerk_bars = self.ax3.bar(df['datetime'], df['jerk_value'], width=0.01, alpha=0.7)

        # Color code jerk bars (red if above threshold)
        for i, bar in enumerate(jerk_bars):
            if df['jerk_value'].iloc[i] > 1.5:  # Threshold value
                bar.set_color('red')
            else:
                bar.set_color('blue')

        # Re-add threshold line
        self.ax3.axhline(y=1.5, color='r', linestyle='--', alpha=0.7)
        if not df.empty:
            self.ax3.text(df['datetime'].iloc[0], 1.6, 'Threshold', color='r')

        # Format the plots
        self.ax1.set_ylabel('Temperature (°C)')
        self.ax1.set_title('Temperature')
        self.ax1.grid(True)

        self.ax2.set_ylabel('Pressure (hPa)')
        self.ax2.set_title('Pressure')
        self.ax2.grid(True)

        self.ax3.set_ylabel('Jerk (m/s³)')
        self.ax3.set_title('Jerk Detection')
        self.ax3.grid(True)

        # Format x-axis to show time nicely
        for ax in [self.ax1, self.ax2, self.ax3]:
            if not df.empty:
                ax.set_xlim(df['datetime'].iloc[0], df['datetime'].iloc[-1])
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        self.fig.tight_layout()
        return self.ax1, self.ax2, self.ax3

    def export_to_csv(self):
        """Export all data to CSV file"""
        filename = f"helmet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
        conn.close()

        df.to_csv(filename, index=False)
        messagebox.showinfo("Export Complete", f"Data exported to {filename}")

    def clear_database(self):
        """Clear all data from database"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all data from the database?"):
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sensor_data")
            conn.commit()
            conn.close()
            messagebox.showinfo("Clear Complete", "All data has been cleared from the database")

    def run_lcs_analysis(self):
        """Run LCS analysis on collected data"""
        # Get recent jerk data
        df = self.get_recent_data(100)
        if df.empty:
            self.results_text.insert(tk.END, "Not enough data for LCS analysis\n")
            return

        jerk_data = df['jerk_value'].tolist()

        # Run LCS analysis against reference patterns
        results = analyze_jerk_patterns(jerk_data, self.reference_patterns)

        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Movement Analysis Results:\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n")

        for pattern_name, result in results.items():
            self.results_text.insert(tk.END, f"Pattern: {pattern_name}\n")
            self.results_text.insert(tk.END, f"Match Percentage: {result['match_percentage']:.2f}%\n")
            self.results_text.insert(tk.END, f"Common Sequence Length: {len(result['common_sequence'])}\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")

        # Check for anomalies
        if len(jerk_data) > 50:
            historical_data = jerk_data[:50]  # First half as reference
            current_data = jerk_data[50:]  # Second half to check

            anomaly_score = detect_anomalies_with_lcs(current_data, historical_data)
            self.results_text.insert(tk.END, f"Anomaly Score: {anomaly_score:.4f}\n")
            if anomaly_score > 0.5:
                self.results_text.insert(tk.END, "WARNING: Unusual pattern detected!\n")

    def run_merge_sort_analysis(self):
        """Run Merge Sort analysis on collected data"""
        # Get recent data
        df = self.get_recent_data(100)
        if df.empty:
            self.results_text.insert(tk.END, "Not enough data for Merge Sort analysis\n")
            return

        # Extract data series
        temp_data = df['temperature'].tolist()
        pressure_data = df['pressure'].tolist()
        jerk_data = df['jerk_value'].tolist()

        # Run Merge Sort analysis
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Database Analysis Results:\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n")

        # Find median values
        sorted_temp = merge_sort(temp_data)
        sorted_pressure = merge_sort(pressure_data)
        sorted_jerk = merge_sort(jerk_data)

        temp_median = sorted_temp[len(sorted_temp) // 2] if len(sorted_temp) % 2 == 1 else (sorted_temp[
                                                                                                len(sorted_temp) // 2 - 1] +
                                                                                            sorted_temp[
                                                                                                len(sorted_temp) // 2]) / 2
        pressure_median = sorted_pressure[len(sorted_pressure) // 2] if len(sorted_pressure) % 2 == 1 else (
                                                                                                                       sorted_pressure[
                                                                                                                           len(sorted_pressure) // 2 - 1] +
                                                                                                                       sorted_pressure[
                                                                                                                           len(sorted_pressure) // 2]) / 2
        jerk_median = sorted_jerk[len(sorted_jerk) // 2] if len(sorted_jerk) % 2 == 1 else (sorted_jerk[
                                                                                                len(sorted_jerk) // 2 - 1] +
                                                                                            sorted_jerk[
                                                                                                len(sorted_jerk) // 2]) / 2

        self.results_text.insert(tk.END, f"Temperature Median: {temp_median:.2f}°C\n")
        self.results_text.insert(tk.END, f"Pressure Median: {pressure_median:.2f}hPa\n")
        self.results_text.insert(tk.END, f"Jerk Median: {jerk_median:.2f}m/s³\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n")

        # Remove outliers and calculate statistics
        if len(sorted_jerk) > 10:
            low_idx = int(len(sorted_jerk) * 0.05)
            high_idx = int(len(sorted_jerk) * 0.95)
            filtered_jerk = sorted_jerk[low_idx:high_idx]

            filtered_median = filtered_jerk[len(filtered_jerk) // 2] if len(filtered_jerk) % 2 == 1 else (filtered_jerk[
                                                                                                              len(filtered_jerk) // 2 - 1] +
                                                                                                          filtered_jerk[
                                                                                                              len(filtered_jerk) // 2]) / 2

            self.results_text.insert(tk.END, f"Filtered Jerk Median (without outliers): {filtered_median:.2f}m/s³\n")
            self.results_text.insert(tk.END, f"Outliers Removed: {len(sorted_jerk) - len(filtered_jerk)}\n")

    def analyze_algorithm_performance(self):
        """Analyze and compare algorithm performance"""
        import time
        import random

        # Create test data of different sizes
        test_sizes = [10, 100, 1000, 5000]
        lcs_times = []
        merge_sort_times = []

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Algorithm Performance Analysis:\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n")

        for size in test_sizes:
            # Generate random data
            data1 = [random.uniform(0, 10) for _ in range(size)]
            data2 = [random.uniform(0, 10) for _ in range(size)]

            # Measure LCS performance
            start_time = time.time()
            lcs(data1, data2)
            lcs_time = time.time() - start_time
            lcs_times.append(lcs_time)

            # Measure Merge Sort performance
            start_time = time.time()
            merge_sort(data1)
            merge_sort_time = time.time() - start_time
            merge_sort_times.append(merge_sort_time)

            self.results_text.insert(tk.END, f"Size: {size}\n")
            self.results_text.insert(tk.END, f"  LCS Time: {lcs_time:.6f} seconds\n")
            self.results_text.insert(tk.END, f"  Merge Sort Time: {merge_sort_time:.6f} seconds\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")

        # Create performance visualization
        self.visualize_performance(test_sizes, lcs_times, merge_sort_times)

    def visualize_performance(self, sizes, lcs_times, merge_sort_times):
        """Visualize algorithm performance comparison"""
        # Create a new window for visualization
        vis_window = tk.Toplevel(self.root)
        vis_window.title("Algorithm Performance Comparison")
        vis_window.geometry("800x600")

        # Create matplotlib figure
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Plot performance data
        ax.plot(sizes, lcs_times, 'bo-', label='LCS Algorithm')
        ax.plot(sizes, merge_sort_times, 'ro-', label='Merge Sort Algorithm')

        ax.set_title('Algorithm Performance Comparison')
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.legend()
        ax.grid(True)

        # Use logarithmic scale for better visualization
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=vis_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_closing(self):
        """Handle window close event"""
        if self.connected:
            self.data_collection_active = False
            time.sleep(1)  # Give collection thread time to stop
        self.root.destroy()


# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetMonitorApp(root)
    root.mainloop()
