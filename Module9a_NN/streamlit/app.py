import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Neural Networks Lab', layout='wide')
st.title('Interactive Artificial Neural Networks Lab')
st.caption(
    'Explore neurons, network structure, and learning through compact diagrams'
    ' and editable numeric values.'
)
st.markdown(
    '**How to use:** Enter values in the small boxes beside the network. The'
    ' diagram, intermediate calculations, and outputs update when you press'
    ' Enter or move to another field.'
)


# ---------- helpers ----------
def number(label, default, key, help=None):
  raw = st.text_input(
      label, value=str(default), key=key, help=help, label_visibility='visible'
  )
  try:
    value = float(raw)
    if not np.isfinite(value):
      raise ValueError
    return value
  except ValueError:
    st.caption(
        f'Enter a valid number for {label}. Using {default} until corrected.'
    )
    return float(default)


def sigmoid(z):
  return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def act(name, z):
  if name == 'Identity':
    return z
  if name == 'Binary step':
    return float(z >= 0)
  if name == 'Sigmoid':
    return float(sigmoid(z))
  if name == 'Tanh':
    return float(np.tanh(z))
  if name == 'ReLU':
    return max(0, z)
  if name == 'Leaky ReLU':
    return z if z >= 0 else 0.1 * z
  if name == 'Softplus':
    return float(np.logaddexp(0, z))
  if name == 'Arctan':
    return float(np.arctan(z))
  return z


def metric_output(label, val):
  st.markdown(
      f"""<div style="border:2px solid #269b68;border-radius:14px;padding:12px;text-align:center;background:rgba(38,155,104,.09)"><div style="font-size:12px;color:gray">{label}</div><div style="font-size:30px;font-weight:700;color:#269b68">{val}</div></div>""",
      unsafe_allow_html=True,
  )


def network_plot(nodes, edges, height=300):
  fig = go.Figure()
  for x1, y1, x2, y2, label in edges:
    fig.add_trace(
        go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode='lines',
            line=dict(color='#9299a3', width=2),
            hoverinfo='text',
            text=[label, label],
            showlegend=False,
        )
    )
  for x, y, label, color in nodes:
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            text=[label],
            textposition='middle center',
            marker=dict(size=48, color=color, line=dict(width=2, color='white')),
            textfont=dict(color='white', size=12),
            showlegend=False,
        )
    )
  fig.update_layout(
      height=height,
      margin=dict(l=5, r=5, t=5, b=5),
      xaxis=dict(visible=False, range=[-.2, 3.2]),
      yaxis=dict(visible=False, range=[-1.4, 1.4]),
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
  )
  st.plotly_chart(
      fig, use_container_width=True, config={'displayModeBar': False}
  )


T = st.tabs([
    '1. Single Neuron',
    '2. Logic Perceptrons',
    '3. XOR Network',
    '4. ANN Forward Pass',
    '5. MSE Loss',
    '6. Gradient Descent',
    '7. Backpropagation',
    '8. Activation Functions',
    '9. Digit Recognition',
])

# 1 single neuron
with T[0]:
  st.header('A single artificial neuron')
  st.latex(r'z=w_1x_1+w_2x_2+b,\qquad y=f(z)')
  left, center, right = st.columns([1.0, 1.4, 1.0], gap='medium')
  with left:
    st.subheader('Inputs and weights')
    x1 = number('Input x₁', 1.0, 'sn_x1')
    w1 = number('Weight w₁', 0.7, 'sn_w1')
    x2 = number('Input x₂', 0.5, 'sn_x2')
    w2 = number('Weight w₂', 0.6, 'sn_w2')
    bias = number('Bias b', -0.5, 'sn_b')
  z = w1 * x1 + w2 * x2 + bias
  with center:
    st.subheader('Neuron structure')
    fn = st.selectbox(
        'Activation function',
        ['Binary step', 'Sigmoid', 'Tanh', 'ReLU', 'Identity'],
        key='sn_fn',
    )
    y = act(fn, z)
    nodes = [
        (0, 0.62, f'x₁\n{x1:g}', '#3478b8'),
        (0, -0.62, f'x₂\n{x2:g}', '#3478b8'),
        (1.55, 0, 'Σ', '#df8732'),
        (2.55, 0, f'y\n{y:.3f}', '#269b68'),
    ]
    edges = [
        (0, 0.62, 1.3, 0.08, f'w₁={w1:g}; contribution={w1*x1:.3f}'),
        (0, -0.62, 1.3, -0.08, f'w₂={w2:g}; contribution={w2*x2:.3f}'),
        (1.8, 0, 2.3, 0, f'{fn}(z)'),
    ]
    network_plot(nodes, edges, 220)
    st.caption(
        f'Connection 1: {w1:g} × {x1:g} = **{w1*x1:.3f}**  |  Connection 2:'
        f' {w2:g} × {x2:g} = **{w2*x2:.3f}**  |  Bias: **{bias:.3f}**'
    )

    st.markdown('**Activation Function Graph**')
    xx = np.linspace(-6, 6, 300)
    yy = [act(fn, float(v)) for v in xx]
    fig_act = go.Figure(
        go.Scatter(x=xx, y=yy, mode='lines', name='f(z)', line=dict(width=2))
    )
    fig_act.add_vline(x=z, line_dash='dash', line_color='red')
    fig_act.add_trace(
        go.Scatter(
            x=[z],
            y=[y],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Current z',
        )
    )
    fig_act.update_layout(
        height=180,
        margin=dict(l=5, r=5, t=10, b=5),
        xaxis_title='z',
        yaxis_title='f(z)',
        showlegend=False,
    )
    st.plotly_chart(fig_act, use_container_width=True)
  with right:
    st.subheader('Neuron output')
    metric_output('OUTPUT y', f'{y:.4f}')
    st.metric('Weighted sum z', f'{z:.4f}')
    st.markdown('**Calculation**')
    st.latex(rf'z=({w1:g})({x1:g})+({w2:g})({x2:g})+({bias:g})={z:.3f}')
    st.latex(rf'y={fn}(z)={y:.4f}')
    st.caption(
        'For Binary step, the output is 1 when z ≥ 0; otherwise it is 0.'
    )

# 2 logic
with T[1]:
  st.header('Perceptrons as logic gates')
  st.write(
      'Choose a linearly separable gate. Edit weights and bias to see how'
      ' each input combination propagates through the network.'
  )
  gate = st.selectbox(
      'Starting example', ['AND', 'OR', 'NAND', 'NOR'], key='lg_gate'
  )
  presets = {
      'AND': (1, 1, -1.5),
      'OR': (1, 1, -0.5),
      'NAND': (-1, -1, 1.5),
      'NOR': (-1, -1, 0.5),
  }
  p = presets[gate]
  c1, c2 = st.columns([1, 1.5])
  with c1:
    a = number('Weight w₁', p[0], 'lg_w1')
    b = number('Weight w₂', p[1], 'lg_w2')
    bias = number('Bias b', p[2], 'lg_b')
    st.latex(
        r'y=\begin{cases}1&w_1x_1+w_2x_2+b\geq0\\0&\text{otherwise}\end{cases}'
    )
  with c2:
    xx, yy = np.meshgrid(
        np.linspace(-0.2, 1.2, 60), np.linspace(-0.2, 1.2, 60)
    )
    zz_grid = a * xx + b * yy + bias
    fig = go.Figure(
        go.Contour(
            x=np.linspace(-0.2, 1.2, 60),
            y=np.linspace(-0.2, 1.2, 60),
            z=zz_grid,
            contours=dict(start=0, end=0, size=1, coloring='lines'),
            showscale=False,
            line=dict(width=3),
        )
    )
    for x1_val, x2_val in [(0, 0), (0, 1), (1, 0), (1, 1)]:
      out_val = int((a * x1_val + b * x2_val + bias) >= 0)
      fig.add_trace(
          go.Scatter(
              x=[x1_val],
              y=[x2_val],
              mode='markers+text',
              text=[str(out_val)],
              textposition='top center',
              marker=dict(
                  size=16, color='#269b68' if out_val else '#d9534f'
              ),
              showlegend=False,
          )
      )
    fig.update_layout(
        height=280,
        margin=dict(l=5, r=5, t=15, b=5),
        xaxis_title='x₁',
        yaxis_title='x₂',
        title='Decision boundary',
    )
    st.plotly_chart(fig, use_container_width=True)

  st.subheader('Network evaluations for all input combinations')
  for x1_val, x2_val in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    zz = a * x1_val + b * x2_val + bias
    out_val = int(zz >= 0)
    col_net, col_info = st.columns([1.5, 1])
    with col_net:
      nodes = [
          (0, 0.62, f'x₁\n{x1_val}', '#3478b8'),
          (0, -0.62, f'x₂\n{x2_val}', '#3478b8'),
          (1.55, 0, 'Σ', '#df8732'),
          (2.55, 0, f'y\n{out_val}', '#269b68' if out_val else '#d9534f'),
      ]
      edges = [
          (0, 0.62, 1.3, 0.08, f'w₁={a:g}'),
          (0, -0.62, 1.3, -0.08, f'w₂={b:g}'),
          (1.8, 0, 2.3, 0, 'step(z)'),
      ]
      network_plot(nodes, edges, 180)
    with col_info:
      st.markdown(f'**Inputs:** $x_1 = {x1_val}, x_2 = {x2_val}$')
      st.latex(
          rf'z = ({a:g})({x1_val}) + ({b:g})({x2_val}) + ({bias:g}) = {zz:.3f}'
      )
      st.latex(rf'y = \text{{step}}(z) = {out_val}')
    st.divider()

  st.info(
      'A single perceptron can represent AND and OR, but not XOR because XOR is'
      ' not linearly separable.'
  )

# 3 XOR (Enhanced with customizable weights and feature space plot)
with T[2]:
  st.header('Why XOR needs a hidden layer')
  st.latex(
      r'h_1=\sigma(w_{11}x_1+w_{12}x_2+b_1),\quad'
      r' h_2=\sigma(w_{21}x_1+w_{22}x_2+b_2)'
  )
  st.latex(r'y=\sigma(v_1h_1+v_2h_2+b_o)')

  col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
  with col_cfg1:
    st.subheader('Hidden Neuron 1')
    w11 = number('Weight x₁->h₁', 20.0, 'xor_w11')
    w12 = number('Weight x₂->h₁', 20.0, 'xor_w12')
    b1 = number('Bias b₁', -10.0, 'xor_b1')
  with col_cfg2:
    st.subheader('Hidden Neuron 2')
    w21 = number('Weight x₁->h₂', -20.0, 'xor_w21')
    w22 = number('Weight x₂->h₂', -20.0, 'xor_w22')
    b2 = number('Bias b₂', 30.0, 'xor_b2')
  with col_cfg3:
    st.subheader('Output Layer')
    v1 = number('Weight h₁->y', 20.0, 'xor_v1')
    v2 = number('Weight h₂->y', 20.0, 'xor_v2')
    bo = number('Bias b_o', -30.0, 'xor_bo')

  st.divider()
  c1, c2, c3 = st.columns([1, 1, 1.5])
  with c1:
    x1 = int(number('Input x₁ (0 or 1)', 0, 'xor_x1'))
    x1 = int(x1 != 0)
  with c2:
    x2 = int(number('Input x₂ (0 or 1)', 1, 'xor_x2'))
    x2 = int(x2 != 0)

  h1 = float(sigmoid(w11 * x1 + w12 * x2 + b1))
  h2 = float(sigmoid(w21 * x1 + w22 * x2 + b2))
  out = float(sigmoid(v1 * h1 + v2 * h2 + bo))

  with c3:
    nodes = [
        (0, 0.55, f'x₁={x1}', '#3478b8'),
        (0, -0.55, f'x₂={x2}', '#3478b8'),
        (1.35, 0.55, f'h₁\n{h1:.2f}', '#df8732'),
        (1.35, -0.55, f'h₂\n{h2:.2f}', '#df8732'),
        (2.65, 0, f'y\n{out:.3f}', '#269b68'),
    ]
    edges = [
        (0, 0.55, 1.1, 0.55, f'{w11:g}, {w12:g}'),
        (0, -0.55, 1.1, 0.55, f'{w11:g}, {w12:g}'),
        (0, 0.55, 1.1, -0.55, f'{w21:g}, {w22:g}'),
        (0, -0.55, 1.1, -0.55, f'{w21:g}, {w22:g}'),
        (1.6, 0.55, 2.4, 0, f'{v1:g}'),
        (1.6, -0.55, 2.4, 0, f'{v2:g}'),
    ]
    network_plot(nodes, edges, 290)

  st.metric('Predicted XOR class', int(out >= 0.5))

  table = []
  for a_val, b_val in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    u = sigmoid(w11 * a_val + w12 * b_val + b1)
    v = sigmoid(w21 * a_val + w22 * b_val + b2)
    q = sigmoid(v1 * u + v2 * v + bo)
    table.append({
        'x₁': a_val,
        'x₂': b_val,
        'h₁': round(u, 3),
        'h₂': round(v, 3),
        'y': round(q, 3),
        'Class': int(q >= 0.5),
    })
  st.dataframe(table, use_container_width=True, hide_index=True)

# 4 forward pass
with T[3]:
  st.header('ANN architecture and forward propagation')
  st.latex(r' z_j=\sum_i w_{ij}x_i+b_j,\qquad a_j=f(z_j)')
  st.write(
      'Edit the values next to each connection. Follow the inputs through two'
      ' hidden neurons and one output neuron.'
  )
  left, mid, right = st.columns([1, 1.25, 1], gap='medium')
  with left:
    st.subheader('Inputs')
    x1 = number('x₁', 0.6, 'fp_x1')
    x2 = number('x₂', 0.2, 'fp_x2')
    st.subheader('Input -> hidden weights')
    w11 = number('x₁ -> h₁', 1.0, 'fp_w11')
    w21 = number('x₂ -> h₁', 0.5, 'fp_w21')
    w12 = number('x₁ -> h₂', -1.0, 'fp_w12')
    w22 = number('x₂ -> h₂', 1.5, 'fp_w22')
    bh1 = number('Hidden bias b₁', 0, 'fp_bh1')
    bh2 = number('Hidden bias b₂', 0, 'fp_bh2')
  h1 = float(sigmoid(w11 * x1 + w21 * x2 + bh1))
  h2 = float(sigmoid(w12 * x1 + w22 * x2 + bh2))
  with mid:
    st.subheader('Network activity')
    nodes = [
        (0, 0.55, f'x₁\n{x1:g}', '#3478b8'),
        (0, -0.55, f'x₂\n{x2:g}', '#3478b8'),
        (1.4, 0.55, f'h₁\n{h1:.3f}', '#df8732'),
        (1.4, -0.55, f'h₂\n{h2:.3f}', '#df8732'),
        (2.8, 0, 'Output', '#269b68'),
    ]
    edges = [
        (0, 0.55, 1.15, 0.55, f'{w11:g}'),
        (0, -0.55, 1.15, 0.55, f'{w21:g}'),
        (0, 0.55, 1.15, -0.55, f'{w12:g}'),
        (0, -0.55, 1.15, -0.55, f'{w22:g}'),
    ]
    network_plot(nodes, edges, 270)
    st.caption(
        f'h₁ = σ({w11:g}·{x1:g} + {w21:g}·{x2:g} + {bh1:g}) = **{h1:.4f}**'
    )
    st.caption(
        f'h₂ = σ({w12:g}·{x1:g} + {w22:g}·{x2:g} + {bh2:g}) = **{h2:.4f}**'
    )
  with right:
    st.subheader('Hidden -> output')
    v1 = number('h₁ -> output', 1.2, 'fp_v1')
    v2 = number('h₂ -> output', -0.7, 'fp_v2')
    bo = number('Output bias', 0, 'fp_bo')
    zo = v1 * h1 + v2 * h2 + bo
    out = float(sigmoid(zo))
    metric_output('NETWORK OUTPUT', f'{out:.4f}')
    st.latex(rf'z_o=({v1:g})({h1:.3f})+({v2:g})({h2:.3f})+({bo:g})={zo:.3f}')
    st.latex(rf'y=\sigma(z_o)={out:.4f}')

# 5 MSE
with T[4]:
  st.header('Loss function: Mean Squared Error')
  st.latex(r'MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2')
  st.write(
      'Enter predicted outputs. Compare them with the target values and see how'
      ' each squared error contributes to the total loss.'
  )
  targets = [0, 1, 1, 0, 1]
  defaults = [0.1, 0.8, 0.6, 0.4, 0.9]
  cols = st.columns(5)
  preds = []
  for i, col in enumerate(cols):
    with col:
      preds.append(number(f'Prediction {i+1}', defaults[i], f'mse_p{i}'))
  errs = [(targets[i] - preds[i]) ** 2 for i in range(5)]
  mse = float(np.mean(errs))
  st.metric('Mean Squared Error', f'{mse:.5f}')
  st.dataframe(
      [{
          'Example': i + 1,
          'Target': targets[i],
          'Prediction': round(preds[i], 4),
          'Squared error': round(errs[i], 5),
      } for i in range(5)],
      use_container_width=True,
      hide_index=True,
  )
  fig = go.Figure()
  fig.add_trace(go.Bar(x=list(range(1, 6)), y=targets, name='Target'))
  fig.add_trace(go.Bar(x=list(range(1, 6)), y=preds, name='Prediction'))
  fig.update_layout(
      barmode='group',
      height=300,
      margin=dict(l=5, r=5, t=20, b=5),
      xaxis_title='Example',
      yaxis_title='Output',
  )
  st.plotly_chart(fig, use_container_width=True)

# 6 gradient descent
with T[5]:
  st.header('Gradient descent and learning rate')
  st.latex(r'w\leftarrow w-\eta\frac{dE}{dw}')
  st.write(
      'For this demonstration, the cost is E(w) = (w − 1)² + 1, so its minimum'
      ' is at w = 1.'
  )
  c1, c2, c3 = st.columns(3)
  with c1:
    eta = number('Learning rate η', 0.15, 'gd_eta')
  with c2:
    start = number('Starting weight', 4, 'gd_start')
  with c3:
    steps_raw = number('Number of update steps', 15, 'gd_steps')
    steps = max(1, min(100, int(steps_raw)))
  path = [start]
  w_val = start
  for _ in range(steps):
    w_val = w_val - eta * 2 * (w_val - 1)
    path.append(w_val)
  grid = np.linspace(min(-5, start - 1), max(5, start + 1), 400)
  fig = go.Figure()
  fig.add_trace(
      go.Scatter(x=grid, y=(grid - 1) ** 2 + 1, name='Cost function')
  )
  fig.add_trace(
      go.Scatter(
          x=path,
          y=(np.array(path) - 1) ** 2 + 1,
          mode='lines+markers',
          name='Update path',
          marker=dict(size=7),
      )
  )
  fig.update_layout(
      height=340,
      xaxis_title='Weight w',
      yaxis_title='Cost E(w)',
      title='Each dot is one parameter update',
  )
  st.plotly_chart(fig, use_container_width=True)
  a, b, c = st.columns(3)
  a.metric('Final weight', f'{path[-1]:.5f}')
  b.metric('Final cost', f'{(path[-1]-1)**2+1:.5f}')
  c.metric('Distance from minimum', f'{abs(path[-1]-1):.5f}')
  st.dataframe(
      [{'Step': i, 'Weight': round(v, 5), 'Cost': round((v - 1) ** 2 + 1, 5)} for i, v in enumerate(path)],
      use_container_width=True,
      hide_index=True,
  )

# 7 backpropagation
with T[6]:
  st.header('Backpropagation: error flows backward')
  st.latex(r'\delta_o=o(1-o)(t-o),\quad \delta_h=h(1-h)w_o\delta_o')
  st.latex(r'\Delta w=\eta\delta\,input,\qquad w_{new}=w+\Delta w')
  st.write(
      'A compact network with one input, one hidden neuron, and one output.'
      ' Edit the values to see the forward and backward calculations.'
  )
  left, right = st.columns([1, 1.3])
  with left:
    x = number('Input x', 0.8, 'bp_x')
    target = number('Target t', 1, 'bp_t')
    eta = number('Learning rate η', 0.5, 'bp_eta')
    wi = number('Input -> hidden weight', 1, 'bp_wi')
    bh = number('Hidden bias', 0, 'bp_bh')
    wo = number('Hidden -> output weight', 1, 'bp_wo')
    bo = number('Output bias', 0, 'bp_bo')

  h = float(sigmoid(wi * x + bh))
  o = float(sigmoid(wo * h + bo))
  do = o * (1 - o) * (target - o)
  dh = h * (1 - h) * wo * do
  dwo = eta * do * h
  dbo = eta * do
  dwi = eta * dh * x
  dbh = eta * dh

  with right:
    nodes = [
        (0, 0, f'x\n{x:g}', '#3478b8'),
        (1.25, 0, f'h\n{h:.3f}', '#df8732'),
        (2.5, 0, f'o\n{o:.3f}', '#269b68'),
    ]
    edges = [
        (0.2, 0, 1.0, 0, f'wᵢ={wi:g}'),
        (1.45, 0, 2.25, 0, f'wₒ={wo:g}'),
    ]
    network_plot(nodes, edges, 200)
    metric_output('PREDICTED OUTPUT (o)', f'{o:.4f}')

    st.markdown('### Step-by-Step Breakdown')
    col_fwd, col_bwd = st.columns(2)
    with col_fwd:
      st.markdown('**Forward Pass**')
      st.latex(rf'h = \sigma({wi:g} \times {x:g} + {bh:g}) = {h:.4f}')
      st.latex(rf'o = \sigma({wo:g} \times {h:.4f} + {bo:g}) = {o:.4f}')
      st.latex(rf'E = \frac{{1}}{{2}}(t - o)^2 = {0.5*(target-o)**2:.5f}')
    with col_bwd:
      st.markdown('**Backward Deltas**')
      st.latex(rf'\delta_o = {do:.4f}')
      st.latex(rf'\delta_h = {dh:.4f}')

    st.markdown('**Parameter Update Summary**')
    st.dataframe(
        [
            {
                'Parameter': 'w_input',
                'Before': wi,
                'Change': round(dwi, 5),
                'After': round(wi + dwi, 5),
            },
            {
                'Parameter': 'b_hidden',
                'Before': bh,
                'Change': round(dbh, 5),
                'After': round(bh + dbh, 5),
            },
            {
                'Parameter': 'w_output',
                'Before': wo,
                'Change': round(dwo, 5),
                'After': round(wo + dwo, 5),
            },
            {
                'Parameter': 'b_output',
                'Before': bo,
                'Change': round(dbo, 5),
                'After': round(bo + dbo, 5),
            },
        ],
        hide_index=True,
        use_container_width=True,
    )

# 8 activation
with T[7]:
  st.header('Activation and transfer functions')
  fn = st.selectbox(
      'Activation function',
      [
          'Identity',
          'Binary step',
          'Sigmoid',
          'Tanh',
          'Arctan',
          'ReLU',
          'Leaky ReLU',
          'Softplus',
      ],
      key='act_fn',
  )
  z = number('Neuron weighted sum z', 1.0, 'act_z')
  formulas = {
      'Identity': r'f(z)=z,\quad f\prime(z)=1',
      'Binary step': r'f(z)=0\ (z<0),\quad f(z)=1\ (z\geq0)',
      'Sigmoid': (
          r'f(z)=\frac{1}{1+e^{-z}},\quad f\prime(z)=f(z)(1-f(z))'
      ),
      'Tanh': r'f(z)=\tanh(z),\quad f\prime(z)=1-f(z)^2',
      'Arctan': r'f(z)=\tan^{-1}(z),\quad f\prime(z)=\frac{1}{1+z^2}',
      'ReLU': r'f(z)=\max(0,z)',
      'Leaky ReLU': r'f(z)=\max(0.1z,z)',
      'Softplus': r'f(z)=\ln(1+e^z)',
  }
  c1, c2 = st.columns([1, 1.5])
  with c1:
    st.latex(formulas[fn])
    metric_output('ACTIVATION OUTPUT', f'{act(fn, z):.5f}')
  with c2:
    xx = np.linspace(-6, 6, 300)
    yy = [act(fn, float(v)) for v in xx]
    fig = go.Figure(
        go.Scatter(
            x=xx, y=yy, mode='lines', name='f(z)', line=dict(width=3)
        )
    )
    fig.add_vline(x=z, line_dash='dash')
    fig.add_trace(
        go.Scatter(
            x=[z],
            y=[act(fn, z)],
            mode='markers',
            marker=dict(size=12),
            name='Current z',
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=5, r=5, t=20, b=5),
        xaxis_title='z',
        yaxis_title='f(z)',
        title=f'{fn} activation',
    )
    st.plotly_chart(fig, use_container_width=True)

# 9 digit
with T[8]:
  st.header('Handwritten digit recognition: pixels to classes')
  st.write(
      'A real network receives pixel values, processes them through hidden'
      ' layers, and produces ten output scores. Enter ten example output'
      ' scores to see how the output layer selects a class.'
  )
  defaults = [0.05, 0.04, 0.03, 0.02, 0.03, 0.06, 0.08, 0.85, 0.04, 0.05]
  scores = []
  for row in range(2):
    cols = st.columns(5)
    for j, col in enumerate(cols):
      i = row * 5 + j
      with col:
        scores.append(
            max(0, number(f'Score for digit {i}', defaults[i], f'digit_score_{i}'))
        )
  total = sum(scores)
  probs = [v / total for v in scores] if total > 0 else [0.1] * 10
  pred = int(np.argmax(probs))
  st.metric('Predicted digit', pred)
  fig = go.Figure(go.Bar(x=list(range(10)), y=probs))
  fig.update_layout(
      height=300,
      margin=dict(l=5, r=5, t=20, b=5),
      xaxis_title='Digit class',
      yaxis_title='Normalized score',
      title='Output layer',
  )
  st.plotly_chart(fig, use_container_width=True)
  st.caption(
      'These are manually entered illustrative scores, not predictions from a'
      ' trained image classifier.'
  )

st.divider()