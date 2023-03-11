"""Here, I'm trying to come up with framework for modeling streaming data.

Thoughts:
1. We'll want to be able to simulate and process realtime with the same code
1. Models are always available for prediction/description
1. The models should act like independent services.
    * They share no state except clock time.
    * Communicate with small messages.
    * Can function with varying levels of completeness of services they depend on
    * messages are pushed, not pulled. So, if model C depends on A and B, C listens to both A and B and only when it
        receives a message from either one will it update and decide whether to publish. It cannot rely on a guarantee
        that A and B push at the same time.
1. Simulation probably requires a queue of events that the services add to
"""

