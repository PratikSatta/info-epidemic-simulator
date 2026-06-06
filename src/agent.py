from __future__ import annotations
from typing import ClassVar


class Agent:
    """
    Represents a single person in the social network.

    ENCAPSULATION:
        - All attributes are private: _trust, _bias, _influence, _state
        - External code accesses them ONLY via @property descriptors
        - Each setter validates the input before storing
        - This prevents invalid states (e.g., trust=5.0, state="X")

    Attributes (private — do not access directly):
        _node_id   : int           -- NetworkX node index (read-only)
        _trust     : float [0, 1]  -- willingness to accept information
        _bias      : float [-1, 1] -- content-type affinity
        _influence : float [0, 1]  -- normalised PageRank
        _state     : str           -- one of {"S", "E", "I", "R"}
    """

    VALID_STATES: ClassVar[frozenset] = frozenset({"S", "E", "I", "R"})

    def __init__(
        self,
        node_id:   int,
        trust:     float,
        bias:      float,
        influence: float,
        state:     str = "S",
    ):
        self._node_id = node_id
        self.trust     = trust      # routes through @trust.setter (validated)
        self.bias      = bias       # routes through @bias.setter
        self.influence = influence  # routes through @influence.setter
        self.state     = state      # routes through @state.setter

    # ── node_id (read-only — never changes after creation) ───────────────────

    @property
    def node_id(self) -> int:
        return self._node_id

    # ── trust property with validation ────────────────────────────────────────

    @property
    def trust(self) -> float:
        return self._trust

    @trust.setter
    def trust(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Trust must be numeric, got {type(value).__name__}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Trust must be in [0.0, 1.0], got {value}")
        self._trust = float(value)

    # ── bias property with validation ─────────────────────────────────────────

    @property
    def bias(self) -> float:
        return self._bias

    @bias.setter
    def bias(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Bias must be numeric, got {type(value).__name__}")
        if not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"Bias must be in [-1.0, 1.0], got {value}")
        self._bias = float(value)

    # ── influence property with validation ────────────────────────────────────

    @property
    def influence(self) -> float:
        return self._influence

    @influence.setter
    def influence(self, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Influence must be in [0.0, 1.0], got {value}")
        self._influence = float(value)

    # ── state property with validation ────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        if value not in self.VALID_STATES:
            raise ValueError(
                f"Invalid state '{value}'. Must be one of {self.VALID_STATES}"
            )
        self._state = value

    # ── Behaviour methods ─────────────────────────────────────────────────────

    def is_infectious(self) -> bool:
        """Return True if this agent is actively spreading information."""
        return self._state == "I"

    def is_susceptible(self) -> bool:
        """Return True if this agent can receive information."""
        return self._state == "S"

    def bias_affinity(self, content_bias: float) -> float:
        """
        How aligned is this agent's bias with the content's bias?
        Returns a value in [0.5, 1.0].
            1.0 = perfect alignment (most likely to accept and share)
            0.5 = opposite bias (least likely to accept)
        """
        return 1.0 - abs(self._bias - content_bias) / 2.0

    def to_dict(self) -> dict:
        """Serialise agent state to a plain dictionary (for logging/export)."""
        return {
            "node_id":   self._node_id,
            "trust":     self._trust,
            "bias":      self._bias,
            "influence": self._influence,
            "state":     self._state,
        }

    def __repr__(self) -> str:
        return (
            f"Agent(id={self._node_id}, state={self._state}, "
            f"trust={self._trust:.2f}, influence={self._influence:.3f})"
        )