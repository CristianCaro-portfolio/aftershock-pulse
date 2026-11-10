# ADR 002: Grid association, and a hybrid forecast instead of a sensor only one

## Association

**Decision.** Triggers become events by grid search: 360 candidate sources over the aftershock
zone, TauP iasp91 P times to each station, an event when at least 3 of 4 stations agree within 2 s
of the moveout of some node.

**Why.** With four stations, a full locator (PyOcto, GaMMA) has little to work with, and the
question is narrower: is this trigger part of the Panama sequence? The grid answers exactly that,
rejects noise at one house (citizen sensors are noisy, AM.RE9F3 sits at 60 times the noise of
CU.BCIP) and events from outside the zone whose moveout fits no node. It is vectorized, about 8 s
for 33,000 triggers.

**Trade off.** Locations are only as good as the 0.1 degree grid and P times from one model.

## Forecast

**What was tried first.** Reasenberg and Jones fitted directly on the sensor catalog, then
extrapolated to M5 with the sensor b value. It forecast 150 to 500 M5 events when 2 to 4 happened.
Two reasons, both measured:

1. The sensor magnitude scale is not mb. Its b value is 0.66 against 1.24 for USGS, so
   extrapolating from M3.2 to M5 on the sensor scale is wrong by an order of magnitude.
2. In the first half day the detector saturates (events overlap), so the early rate is flat and
   the fitted decay is too slow.

**Decision.** The sensor catalog is used only for timing: the decay p and c, fitted after half a
day on events 0.5 above the sensor Mc. Productivity comes from USGS events with the decay fixed,
which has a closed form. Both the USGS fit and the hybrid use the short term completeness after a
large mainshock, Mc(t) = Mm - 4.5 - 0.75 log10(t) (Helmstetter, Kagan and Jackson 2006), so the
first hours count only what the network could see.

**Result.** The hybrid and the USGS only fit tie on mean Poisson log score (-1.98 against -1.99),
both contain every outcome in their 95% interval, and both beat generic California parameters
(-6.63, which over-forecast 2 to 5 times). The denser catalog did not improve the M5 forecast on
this sequence. That is reported as is.
