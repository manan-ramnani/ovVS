# Security

Report vulnerabilities privately. Do not file public issues with exploit details.

This library does not accept untrusted serialized indexes from the network
without validation of magic/version/size fields (`ovvsCagraDeserialize`).
