import { useState } from 'react';
export default function Comparison() {
  const [tokens, setTokens] = useState(8);
  return <section className="panel comparison"><h2>Scaling comparison <b>Presentation only</b></h2><label>Sequence length <input aria-label="Sequence length" type="range" min="3" max="18" value={tokens} onChange={e => setTokens(+e.target.value)}/> {tokens}</label><div className="compare-grid"><div><strong>Standard self-attention</strong><div className="pairwise" style={{ gridTemplateColumns: `repeat(${tokens}, 1fr)` }}>{Array.from({ length: tokens * tokens }, (_, i) => <i key={i}/>)}</div><small>{tokens}² = {tokens * tokens} pairwise score cells</small></div><div><strong>Linear attention</strong><div className="fixed-state">S<br/><small>d_feature × d_value</small><br/>Z</div><small>Fixed state shape; recurrence is linear in tokens with dimensions fixed.</small></div></div></section>;
}
