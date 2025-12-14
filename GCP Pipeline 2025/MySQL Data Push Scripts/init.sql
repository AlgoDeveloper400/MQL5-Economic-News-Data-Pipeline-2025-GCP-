-- ============================================
--   Initialize MySQL Database: forex_events
-- ============================================

-- Create database
CREATE DATABASE IF NOT EXISTS forex_events;
USE forex_events;

-- Main events table (keeping DATE type for proper date operations)
CREATE TABLE IF NOT EXISTS events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Date DATE NOT NULL,
    Time TIME NOT NULL,
    Currency VARCHAR(10) NOT NULL,
    Event VARCHAR(255) NOT NULL,
    Impact VARCHAR(20),
    Actual VARCHAR(50),
    Forecast VARCHAR(50),
    Previous VARCHAR(50),
    UNIQUE KEY unique_event (Date, Time, Currency, Event)
);

-- Create view for consistent date formatting
CREATE OR REPLACE VIEW events_formatted AS
SELECT 
    id,
    DATE_FORMAT(Date, '%e %M %Y') as Date,
    Time,
    Currency,
    Event,
    Impact,
    Actual,
    Forecast,
    Previous
FROM events;

-- Training metrics table
CREATE TABLE IF NOT EXISTS train_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Currency VARCHAR(10) NOT NULL,
    Event VARCHAR(255) NOT NULL,
    R2 FLOAT,
    MSE FLOAT,
    Samples INT
);

-- Validation metrics table
CREATE TABLE IF NOT EXISTS validate_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Currency VARCHAR(10) NOT NULL,
    Event VARCHAR(255) NOT NULL,
    R2 FLOAT,
    MSE FLOAT,
    Samples INT
);

-- Test forecasts table 
CREATE TABLE IF NOT EXISTS test_forecasts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Currency VARCHAR(10) NOT NULL,
    Event VARCHAR(255) NOT NULL,
    R2 FLOAT,
    MSE FLOAT,
    Samples INT
);

-- Live forecasts table 
CREATE TABLE IF NOT EXISTS live_forecasts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Currency VARCHAR(10) NOT NULL,
    Event VARCHAR(255) NOT NULL,
    ForecastValue FLOAT
);