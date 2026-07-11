# ADR-0003: FFN is the Sole Knowledge Carrier

All factual knowledge resides exclusively in FFN weights. SFT transfers format (how to answer), not knowledge (what to know).

Empirical evidence from Nano evaluation: 50-question accuracy was 4%, entity consistency 0%. This proved that 40M parameters are insufficient for factual knowledge storage regardless of training data volume. When scaling to 87M, the decision was to keep layers at 12 and expand `d_ff` from 1365 to 2048 (3.4× FFN capacity).

Consequence: Lite 87M allocates 65% of parameters to FFN. Vocabulary expansion is deprioritized because it steals from the FFN budget.
