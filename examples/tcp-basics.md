# TCP basics

TCP (Transmission Control Protocol) is a connection-oriented transport-layer
protocol. Before any application data is exchanged, the client and server
establish a connection.

## The three-way handshake

1. The client sends a SYN segment with an initial sequence number.
2. The server replies with SYN-ACK, acknowledging the client's sequence
   number and sending its own.
3. The client sends ACK, and the connection is established.

## Reliability

TCP numbers every byte it sends. The receiver acknowledges what it has
received, and the sender retransmits anything that isn't acknowledged
before a timeout. This gives applications an ordered, reliable byte stream.

## Flow control vs congestion control

- Flow control stops a fast sender from overwhelming a slow receiver. The
  receiver advertises a window size telling the sender how much it can accept.
- Congestion control stops senders from overwhelming the network. TCP starts
  slowly (slow start) and backs off when it detects packet loss.

## Closing a connection

Either side can close its half of the connection by sending FIN, which the
other side acknowledges. A full close usually takes four segments:
FIN, ACK, FIN, ACK.

## TCP vs UDP

UDP is connectionless and has no handshake, acknowledgements or
retransmission. It is faster and simpler, which suits streaming, gaming and
DNS lookups, where an occasional lost packet matters less than delay.
