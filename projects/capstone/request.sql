SELECT BillingCountry country, datetime(InvoiceDate, 'start of month') date_month, sum(Total) total_month
FROM Invoice
GROUP BY country
ORDER BY country, date_month



