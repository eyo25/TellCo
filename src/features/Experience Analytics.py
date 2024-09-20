# Function to compute top, bottom, and most frequent values
def compute_top_bottom_frequent(df, column):
    """Return top 10, bottom 10, and most frequent values for a given column."""
    top_10 = df.nlargest(10, column)[[column]]
    bottom_10 = df.nsmallest(10, column)[[column]]
    most_frequent = df[column].value_counts().head(10)

    return top_10, bottom_10, most_frequent

# Top, bottom, and most frequent for TCP retransmission
tcp_top, tcp_bottom, tcp_frequent = compute_top_bottom_frequent(df, 'TCP DL Retrans. Vol (Bytes)')
# Top, bottom, and most frequent for RTT
rtt_top, rtt_bottom, rtt_frequent = compute_top_bottom_frequent(df, 'Avg RTT DL (ms)')
# Top, bottom, and most frequent for Throughput
throughput_top, throughput_bottom, throughput_frequent = compute_top_bottom_frequent(df, 'Avg Bearer TP DL (kbps)')

# Display results
print("Top 10 TCP Retransmission Values:\n", tcp_top)
print("Bottom 10 TCP Retransmission Values:\n", tcp_bottom)
print("Most Frequent TCP Retransmission Values:\n", tcp_frequent)

print("Top 10 RTT Values:\n", rtt_top)
print("Bottom 10 RTT Values:\n", rtt_bottom)
print("Most Frequent RTT Values:\n", rtt_frequent)

print("Top 10 Throughput Values:\n", throughput_top)
print("Bottom 10 Throughput Values:\n", throughput_bottom)
print("Most Frequent Throughput Values:\n", throughput_frequent)
