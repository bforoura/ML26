import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Interactive Neural Networks Lab", page_icon="🧠", layout="wide")

st.title("🧠 Interactive Artificial Neural Networks Lab")
st.caption("A classroom demonstration companion to the Artificial Neural Networks lecture notes.")

st.markdown("""
Explore how an artificial neuron combines inputs, weights, and a bias; how perceptrons implement
logic functions; and how neural networks learn through forward propagation, loss, and backpropagation.
Change the controls and observe the numerical and graphical results update immediately.
""")

def activation(name, z, alpha=0.1):
    z = np.asarray(z, dtype=float)
    if name == "Identity":
        return z
    if name == "Binary step":
        return (z >= 0).astype(float)
    if name == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    if name == "Tanh":
        return np.tanh(z)
    if name == "ReLU":
        return np.maximum(0, z)
    if name == "Leaky ReLU":
        return np.where(z < 0, alpha * z, z)
    if name == "Softplus":
        return np.logaddexp(0, z)
    if name == "Arctan":
        return np.arctan(z)
    return z

def derivative(name, z, alpha=0.1):
    z = np.asarray(z, dtype=float)
    if name == "Identity":
        return np.ones_like(z)
    if name == "Binary step":
        return np.full_like(z, np.nan)
    if name == "Sigmoid":
        a = activation(name, z)
        return a * (1 - a)
    if name == "Tanh":
        a = np.tanh(z)
        return 1 - a**2
    if name == "ReLU":
        return (z > 0).astype(float)
    if name == "Leaky ReLU":
        return np.where(z < 0, alpha, 1.0)
    if name == "Softplus":
        return 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    if name == "Arctan":
        return 1 / (1 + z**2)
    return np.ones_like(z)

def make_activation_plot(name, alpha=0.1):
    x = np.linspace(-6, 6, 500)
    y = activation(name, x, alpha)
    dy = derivative(name, x, alpha)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name="f(x)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=x, y=dy, name="f′(x)", line=dict(dash="dash", width=2), yaxis="y2"))
    fig.update_layout(
        height=390, margin=dict(l=20, r=20, t=35, b=20),
        xaxis_title="Input z", yaxis_title="Activation f(z)",
        yaxis2=dict(title="Derivative f′(z)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12), title=f"{name}: activation and derivative"
    )
    return fig

def neuron_output(inputs, weights, bias, func):
    z = float(np.dot(inputs, weights) + bias)
    return z, float(activation(func, z))

tabs = st.tabs([
    "1. Simple Neuron", "2. Perceptron Logic", "3. XOR Network",
    "4. ANN & Forward Pass", "5. Loss (MSE)", "6. Gradient Descent",
    "7. Backpropagation", "8. Activation Functions", "9. Digit Recognition"
])

# 1. Single neuron
with tabs[0]:
    st.header("Interactive artificial neuron")
    st.markdown(r"""
    Change each input and its synaptic weight in the neuron panel. The weighted signals are
    combined with the bias, then passed through the activation function:

    \[
    z = \sum_{i=1}^{n} w_i x_i + b,\qquad y=f(z)
    \]

    The live output graph shows how the neuron's numerical output changes as an input changes.
    """)
    n_inputs = st.selectbox("Number of input channels", [2, 3, 4], index=2, key="n_inputs")

    defaults_x = [1.0, 0.5, -0.5, 1.0]
    defaults_w = [0.7, 0.6, 1.4, -0.5]
    vals, weights = [], []

    st.subheader("Neuron control panel")
    st.caption("Each row is one incoming signal. Adjust x and w directly in the corresponding input channel.")
    header = st.columns([1.0, 1.0, 1.0, 1.25])
    header[0].markdown("**Input channel**")
    header[1].markdown("**Input xᵢ**")
    header[2].markdown("**Weight wᵢ**")
    header[3].markdown("**Contribution wᵢxᵢ**")
    for i in range(n_inputs):
        row = st.columns([1.0, 1.0, 1.0, 1.25])
        with row[0]:
            st.markdown(f"**x{i+1} → neuron**")
        with row[1]:
            xv = st.slider(f"Input x{i+1}", -2.0, 2.0, defaults_x[i], 0.1,
                           key=f"nx{i}", label_visibility="collapsed")
        with row[2]:
            wv = st.slider(f"Weight w{i+1}", -2.0, 2.0, defaults_w[i], 0.1,
                           key=f"nw{i}", label_visibility="collapsed")
        with row[3]:
            st.metric(f"w{i+1} × x{i+1}", f"{xv*wv:+.2f}")
        vals.append(xv)
        weights.append(wv)

    controls = st.columns([1, 1, 1.5])
    with controls[0]:
        bias = st.slider("Bias b", -4.0, 4.0, -0.5, 0.1, key="nb")
    with controls[1]:
        func = st.selectbox("Activation function",
                            ["Sigmoid", "Tanh", "ReLU", "Identity", "Binary step"],
                            index=0, key="nfunc")
    with controls[2]:
        selected_input = st.selectbox("Sweep this input in the output graph",
                                      [f"x{i+1}" for i in range(n_inputs)], index=0)

    z, y = neuron_output(np.array(vals), np.array(weights), bias, func)
    contributions = np.array(vals) * np.array(weights)

    st.subheader("Signal flow through the neuron")
    # Build a large, clean flow diagram. Input and weight values are repeated here
    # beside the actual sliders above so students can connect each control to its signal.
    diagram = go.Figure()
    y_positions = np.linspace(0.82, 0.18, n_inputs)
    for i, (xv, wv, contrib, yp) in enumerate(zip(vals, weights, contributions, y_positions)):
        line_color = "#2E8B57" if contrib >= 0 else "#D9534F"
        diagram.add_trace(go.Scatter(
            x=[0.12, 0.49], y=[yp, 0.5], mode="lines",
            line=dict(width=2.5 + min(abs(wv), 3), color=line_color),
            hovertemplate=f"Input x{i+1}: {xv:.2f}<br>Weight w{i+1}: {wv:.2f}<br>Contribution: {contrib:.2f}<extra></extra>",
            showlegend=False))
        diagram.add_trace(go.Scatter(
            x=[0.12], y=[yp], mode="markers+text",
            marker=dict(size=28, color="#4C78A8", line=dict(color="white", width=2)),
            text=[f"x{i+1} = {xv:.1f}"], textposition="middle left",
            textfont=dict(size=13), showlegend=False))
        diagram.add_annotation(x=0.30, y=(yp+0.5)/2, text=f"w{i+1}={wv:.1f}",
                               showarrow=False, bgcolor="white", bordercolor=line_color,
                               borderwidth=1, font=dict(size=11))
    diagram.add_shape(type="circle", x0=0.43, y0=0.35, x1=0.62, y1=0.65,
                      line=dict(width=3, color="#F58518"), fillcolor="#FDE6CC")
    diagram.add_annotation(x=0.525, y=0.54, text="Σ", showarrow=False, font=dict(size=27))
    diagram.add_annotation(x=0.525, y=0.43, text=f"z={z:.2f}", showarrow=False, font=dict(size=12))
    diagram.add_trace(go.Scatter(x=[0.62, 0.86], y=[0.5, 0.5], mode="lines",
                                 line=dict(width=4, color="#54A24B"), showlegend=False))
    diagram.add_annotation(x=0.73, y=0.57, text=f"f(z) = {func}", showarrow=False,
                           font=dict(size=12))
    diagram.add_annotation(x=0.90, y=0.5, text=f"y = {y:.3f}", showarrow=False,
                           xanchor="left", font=dict(size=17, color="#2E8B57"))
    diagram.update_layout(height=360, margin=dict(l=25, r=90, t=20, b=20),
                          xaxis=dict(visible=False, range=[0, 1.15]),
                          yaxis=dict(visible=False, range=[0, 1]),
                          plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(diagram, use_container_width=True, config={"displayModeBar": False})

    metrics = st.columns(3)
    metrics[0].metric("Weighted sum (z)", f"{z:.3f}")
    metrics[1].metric("Neuron output (y)", f"{y:.3f}")
    metrics[2].metric("Total input contribution", f"{contributions.sum():.3f}")

    st.subheader("Live output response")
    sweep_idx = int(selected_input[1:]) - 1
    sweep_values = np.linspace(-2, 2, 200)
    sweep_z = []
    for value in sweep_values:
        temp = vals.copy()
        temp[sweep_idx] = value
        sweep_z.append(float(np.dot(temp, weights) + bias))
    sweep_y = activation(func, np.array(sweep_z))
    current_x = vals[sweep_idx]
    current_y = y
    response_fig = go.Figure()
    response_fig.add_trace(go.Scatter(x=sweep_values, y=sweep_y, mode="lines",
                                      name="Neuron response", line=dict(width=3)))
    response_fig.add_trace(go.Scatter(x=[current_x], y=[current_y], mode="markers+text",
                                      name="Current setting",
                                      marker=dict(size=15, color="#E45756",
                                                  line=dict(color="white", width=2)),
                                      text=[f"Current output = {current_y:.3f}"],
                                      textposition="top center"))
    response_fig.update_layout(height=390, margin=dict(l=25, r=20, t=35, b=35),
                               title=f"Output y versus selected input {selected_input}",
                               xaxis_title=f"Input {selected_input}",
                               yaxis_title="Neuron output y",
                               legend=dict(orientation="h", y=1.12))
    st.plotly_chart(response_fig, use_container_width=True)

    st.subheader("What contributes to the weighted sum?")
    contribution_fig = go.Figure(go.Bar(
        x=[f"w{i+1}x{i+1}" for i in range(n_inputs)] + ["Bias"],
        y=list(contributions) + [bias],
        marker_color=["#54A24B" if v >= 0 else "#E45756"
                      for v in list(contributions) + [bias]],
        text=[f"{v:+.2f}" for v in list(contributions) + [bias]],
        textposition="outside"))
    contribution_fig.update_layout(height=300, title="Individual contributions to z",
                                   yaxis_title="Contribution", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(contribution_fig, use_container_width=True)
    st.info("Teaching idea: Ask students to predict the direction of the output change before moving an input or weight. The response curve makes the effect visible, while the contribution chart explains why it happened.")

# 2 Logic perceptron
with tabs[1]:
    st.header("Perceptrons as logic gates")
    st.markdown("A perceptron can represent linearly separable logic functions such as AND and OR. A single perceptron cannot represent XOR.")
    gate = st.selectbox("Choose a logic function", ["AND", "OR", "NAND", "NOR"])
    if gate == "AND":
        w, b = (1,1), -1.5
    elif gate == "OR":
        w, b = (1,1), -0.5
    elif gate == "NAND":
        w, b = (-1,-1), 1.5
    else:
        w, b = (-1,-1), 0.5
    w1 = st.slider("w1", -3.0, 3.0, float(w[0]), 0.1, key="logicw1")
    w2 = st.slider("w2", -3.0, 3.0, float(w[1]), 0.1, key="logicw2")
    b = st.slider("Bias", -3.0, 3.0, float(b), 0.1, key="logicb")
    rows = []
    for x1,x2 in [(0,0),(0,1),(1,0),(1,1)]:
        z = w1*x1 + w2*x2 + b
        rows.append({"x1":x1,"x2":x2,"z = w1x1+w2x2+b":round(z,2),"Output":int(z>=0)})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    xx, yy = np.meshgrid(np.linspace(-0.2,1.2,100), np.linspace(-0.2,1.2,100))
    zz = w1*xx + w2*yy + b
    fig = go.Figure()
    fig.add_trace(go.Contour(x=np.linspace(-.2,1.2,100), y=np.linspace(-.2,1.2,100), z=zz,
        contours=dict(start=0,end=0,size=1, coloring="lines"), line=dict(width=3), showscale=False, name="Decision boundary"))
    for x1,x2 in [(0,0),(0,1),(1,0),(1,1)]:
        out = int(w1*x1+w2*x2+b >= 0)
        fig.add_trace(go.Scatter(x=[x1],y=[x2],mode="markers+text",text=[str(out)],textposition="top center",
            marker=dict(size=17,color="#54A24B" if out else "#E45756"),name=f"({x1},{x2}) → {out}"))
    fig.update_layout(height=400,xaxis_title="x1",yaxis_title="x2",title="Decision boundary (zero activation)")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("The slider settings determine the displayed gate. The default weights and bias illustrate the selected gate; changing them changes the truth table and boundary.")

# 3 XOR
with tabs[2]:
    st.header("Why XOR needs multiple neurons")
    st.markdown(r"""
    XOR is not linearly separable: no single straight decision boundary can separate its 0s from its 1s.
    The lecture's network uses two hidden neurons to create intermediate decision regions and one output neuron.
    """)
    st.latex(r"h_1=\sigma(20x_1+20x_2-10)")
    st.latex(r"h_2=\sigma(-20x_1-20x_2+30)")
    st.latex(r"y=\sigma(20h_1+20h_2-30)")
    st.markdown("For a clear step-like demonstration, the sigmoid is used with steep weights as in the lecture diagram.")
    x1 = st.select_slider("Input x1", options=[0,1], value=0, key="xorx1")
    x2 = st.select_slider("Input x2", options=[0,1], value=0, key="xorx2")
    sig = lambda q: 1/(1+np.exp(-np.clip(q,-60,60)))
    h1 = sig(20*x1+20*x2-10)
    h2 = sig(-20*x1-20*x2+30)
    y = sig(20*h1+20*h2-30)
    a,b,c = st.columns(3)
    a.metric("Hidden neuron h1",f"{h1:.4f}")
    b.metric("Hidden neuron h2",f"{h2:.4f}")
    c.metric("Output y",f"{y:.4f}")
    st.metric("Predicted XOR class",str(int(y>=0.5)))
    truth = []
    for p,q in [(0,0),(0,1),(1,0),(1,1)]:
        u=sig(20*p+20*q-10); v=sig(-20*p-20*q+30); o=sig(20*u+20*v-30)
        truth.append({"x1":p,"x2":q,"h1":round(u,3),"h2":round(v,3),"y":round(o,3),"Class":int(o>=.5)})
    st.dataframe(truth,use_container_width=True,hide_index=True)
    st.info("The hidden layer transforms the original inputs into a representation that makes the XOR classes separable for the output neuron.")

# 4 ANN
with tabs[3]:
    st.header("ANN architecture and forward propagation")
    st.markdown(r"""
    A feedforward network sends information from the input layer through hidden layers to the output layer.
    Each neuron computes a weighted sum plus bias, then applies an activation function.
    """)
    n_in = st.slider("Input neurons",1,5,2)
    n_hidden = st.slider("Hidden neurons",1,5,3)
    n_out = st.slider("Output neurons",1,3,1)
    fig=go.Figure()
    layers=[n_in,n_hidden,n_out]
    xpositions=[0,1,2]
    coords=[]
    for li,count in enumerate(layers):
        ys=np.linspace(1,-1,count) if count>1 else [0]
        coords.append([(xpositions[li],y) for y in ys])
    for li in range(2):
        for xa,ya in coords[li]:
            for xb,yb in coords[li+1]:
                fig.add_trace(go.Scatter(x=[xa,xb],y=[ya,yb],mode="lines",line=dict(color="#A0A0A0",width=1),hoverinfo="skip",showlegend=False))
    colors=["#4C78A8","#F58518","#54A24B"]
    for li,layer in enumerate(coords):
        fig.add_trace(go.Scatter(x=[p[0] for p in layer],y=[p[1] for p in layer],mode="markers+text",
            text=[f"{['Input','Hidden','Output'][li]} {i+1}" for i in range(len(layer))],
            textposition="middle right",marker=dict(size=30,color=colors[li],line=dict(width=1,color="white")),showlegend=False))
    fig.update_layout(height=430,xaxis=dict(tickvals=[0,1,2],ticktext=["Input layer","Hidden layer","Output layer"],range=[-.3,2.5]),yaxis=dict(visible=False),title="Fully connected feedforward architecture",margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Numerical forward pass: 2 inputs → 2 hidden neurons → 1 output")
    col1,col2=st.columns(2)
    with col1:
        a1=st.slider("Input x1",0.0,1.0,0.6,.05,key="fp_x1")
        a2=st.slider("Input x2",0.0,1.0,0.2,.05,key="fp_x2")
        w11=st.slider("Weight x1 → h1", -3.0,3.0,1.0,.1)
        w21=st.slider("Weight x2 → h1", -3.0,3.0,.5,.1)
        w12=st.slider("Weight x1 → h2", -3.0,3.0,-1.0,.1)
        w22=st.slider("Weight x2 → h2", -3.0,3.0,1.5,.1)
    with col2:
        bh1=st.slider("Bias h1",-2.0,2.0,0.0,.1)
        bh2=st.slider("Bias h2",-2.0,2.0,0.0,.1)
        v1=st.slider("Weight h1 → output",-3.0,3.0,1.2,.1)
        v2=st.slider("Weight h2 → output",-3.0,3.0,-.7,.1)
        bo=st.slider("Output bias",-2.0,2.0,0.0,.1)
    h1v=float(activation("Sigmoid",w11*a1+w21*a2+bh1))
    h2v=float(activation("Sigmoid",w12*a1+w22*a2+bh2))
    out=float(activation("Sigmoid",v1*h1v+v2*h2v+bo))
    st.code(f"z_h1 = {w11:.2f}×{a1:.2f} + {w21:.2f}×{a2:.2f} + {bh1:.2f} = {w11*a1+w21*a2+bh1:.3f}\nh1 = sigmoid(z_h1) = {h1v:.3f}\nz_h2 = {w12:.2f}×{a1:.2f} + {w22:.2f}×{a2:.2f} + {bh2:.2f} = {w12*a1+w22*a2+bh2:.3f}\nh2 = sigmoid(z_h2) = {h2v:.3f}\nz_out = {v1:.2f}×{h1v:.3f} + {v2:.2f}×{h2v:.3f} + {bo:.2f}\ny = sigmoid(z_out) = {out:.3f}")
    st.metric("Network output",f"{out:.4f}")

# 5 MSE
with tabs[4]:
    st.header("Loss function: Mean Squared Error")
    st.latex(r"MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2")
    st.markdown("The expected (target) values are compared with the network's predicted values. Larger errors contribute quadratically more to MSE.")
    targets=np.array([0.0,1.0,1.0,0.0,1.0])
    preds=[]
    cols=st.columns(5)
    for i,col in enumerate(cols):
        with col:
            preds.append(st.slider(f"Prediction {i+1}",0.0,1.0,float([.1,.8,.6,.4,.9][i]),.05,key=f"pred{i}"))
    preds=np.array(preds)
    errors=targets-preds
    mse=float(np.mean(errors**2))
    st.metric("Mean Squared Error",f"{mse:.4f}")
    fig=go.Figure()
    fig.add_trace(go.Bar(x=[f"Example {i+1}" for i in range(5)],y=targets,name="Expected y"))
    fig.add_trace(go.Bar(x=[f"Example {i+1}" for i in range(5)],y=preds,name="Predicted ŷ"))
    fig.update_layout(barmode="group",height=350,yaxis=dict(range=[0,1.1]),title="Expected versus predicted")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe([{"Example":i+1,"Expected y":targets[i],"Predicted ŷ":preds[i],"Squared error":round(errors[i]**2,4)} for i in range(5)],hide_index=True,use_container_width=True)
    st.caption("The lecture also shows a half-squared-error objective, E = ½ Σ(t − o)², in its backpropagation section. This tab uses the conventional mean over N examples.")

# 6 Gradient descent
with tabs[5]:
    st.header("Gradient descent and learning rate")
    st.latex(r"\mathbf{w}\leftarrow\mathbf{w}-\eta\nabla E(\mathbf{w})")
    st.markdown("Gradient descent moves parameters in the direction opposite the gradient. The learning rate η controls the step size.")
    eta=st.slider("Learning rate η",0.001,1.0,0.15,0.001,format="%.3f")
    start=st.slider("Starting weight", -5.0,5.0,4.0,.25)
    steps=st.slider("Number of update steps",1,40,15)
    # Convex cost E(w) = (w-1)^2 + 1, derivative 2(w-1)
    path=[float(start)]
    w=float(start)
    for _ in range(steps):
        w=w-eta*2*(w-1)
        path.append(w)
    grid=np.linspace(-5,5,400)
    cost=(grid-1)**2+1
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=grid,y=cost,name="Cost E(w)",line=dict(width=3)))
    pathcost=(np.array(path)-1)**2+1
    fig.add_trace(go.Scatter(x=path,y=pathcost,mode="lines+markers",name="Optimization path",marker=dict(size=8)))
    fig.update_layout(height=420,xaxis_title="Weight w",yaxis_title="Cost E(w)",title="Gradient descent on E(w) = (w − 1)² + 1")
    st.plotly_chart(fig,use_container_width=True)
    c1,c2,c3=st.columns(3)
    c1.metric("Final weight",f"{path[-1]:.4f}")
    c2.metric("Final cost",f"{pathcost[-1]:.4f}")
    c3.metric("Distance to minimum",f"{abs(path[-1]-1):.4f}")
    st.dataframe([{"Step":i,"Weight":round(v,5),"Cost":round(float((v-1)**2+1),5)} for i,v in enumerate(path)],hide_index=True,use_container_width=True)
    st.warning("This is a one-parameter teaching example, not a trained network. Very large learning rates can overshoot or diverge; small rates make slower progress.")

# 7 Backprop
with tabs[6]:
    st.header("Backpropagation: compute error, then update weights")
    st.markdown("The lecture's algorithm computes output-layer error signals, propagates them to hidden units, and adjusts weights using the learning rate.")
    st.latex(r"\delta_k=o_k(1-o_k)(t_k-o_k)")
    st.latex(r"\delta_h=o_h(1-o_h)\sum_{k\in outputs}w_{hk}\delta_k")
    st.latex(r"\Delta w_{ij}=\eta\,\delta_j\,x_i,\qquad w_{ij}\leftarrow w_{ij}+\Delta w_{ij}")
    st.caption("The equations above use the sigmoid derivative shown in the lecture. This compact example uses one input, one hidden neuron, and one output neuron.")
    c1,c2=st.columns(2)
    with c1:
        x=st.slider("Training input x",0.0,1.0,.8,.05,key="bp_x")
        target=st.slider("Target t",0.0,1.0,1.0,.05,key="bp_t")
        eta_bp=st.slider("Learning rate η",0.01,1.0,.5,.01,key="bp_eta")
    with c2:
        w_in=st.slider("Input → hidden weight", -4.0,4.0,1.0,.1,key="bp_wi")
        b_h=st.slider("Hidden bias", -4.0,4.0,0.0,.1,key="bp_bh")
        w_out=st.slider("Hidden → output weight", -4.0,4.0,1.0,.1,key="bp_wo")
        b_o=st.slider("Output bias", -4.0,4.0,0.0,.1,key="bp_bo")
    sigmoid=lambda q:1/(1+np.exp(-np.clip(q,-60,60)))
    zh=w_in*x+b_h; oh=sigmoid(zh)
    zo=w_out*oh+b_o; oo=sigmoid(zo)
    delta_o=oo*(1-oo)*(target-oo)
    delta_h=oh*(1-oh)*w_out*delta_o
    dwo=eta_bp*delta_o*oh
    dbo=eta_bp*delta_o
    dwi=eta_bp*delta_h*x
    dbh=eta_bp*delta_h
    err=.5*(target-oo)**2
    st.subheader("Forward pass")
    st.code(f"z_h = w_in·x + b_h = {zh:.4f}\nh = sigmoid(z_h) = {oh:.4f}\nz_o = w_out·h + b_o = {zo:.4f}\no = sigmoid(z_o) = {oo:.4f}\nE = ½(t − o)² = {err:.6f}")
    st.subheader("Backward pass")
    st.code(f"δ_output = o(1−o)(t−o) = {delta_o:.6f}\nδ_hidden = h(1−h)·w_out·δ_output = {delta_h:.6f}\nΔw_out = η·δ_output·h = {dwo:.6f}\nΔw_in = η·δ_hidden·x = {dwi:.6f}")
    st.subheader("Updated parameters")
    updated={"Parameter":["w_in","b_h","w_out","b_o"],"Before":[w_in,b_h,w_out,b_o],"Change":[dwi,dbh,dwo,dbo],"After":[w_in+dwi,b_h+dbh,w_out+dwo,b_o+dbo]}
    st.dataframe(updated,hide_index=True,use_container_width=True)
    st.info("Move the target or initial weights. Observe how the output error determines the output delta, which is then passed backward to the hidden neuron.")

# 8 Activations
with tabs[7]:
    st.header("Activation and transfer functions")
    st.markdown("Activation functions introduce nonlinear mappings. The lecture lists Identity, Binary step, Logistic/Sigmoid, Tanh, Arctan, ReLU, PReLU-style, ELU, and Softplus.")
    act=st.selectbox("Activation function",["Identity","Binary step","Sigmoid","Tanh","Arctan","ReLU","Leaky ReLU","Softplus"])
    alpha=st.slider("Negative slope α (Leaky ReLU)",0.01,1.0,0.1,0.01)
    st.plotly_chart(make_activation_plot(act,alpha),use_container_width=True)
    formulas={
        "Identity":r"f(z)=z,\quad f'(z)=1",
        "Binary step":r"f(z)=\begin{cases}0&z<0\\1&z\geq0\end{cases}",
        "Sigmoid":r"f(z)=\frac{1}{1+e^{-z}},\quad f'(z)=f(z)(1-f(z))",
        "Tanh":r"f(z)=\tanh(z),\quad f'(z)=1-f(z)^2",
        "Arctan":r"f(z)=\tan^{-1}(z),\quad f'(z)=\frac{1}{1+z^2}",
        "ReLU":r"f(z)=\max(0,z)",
        "Leaky ReLU":r"f(z)=\begin{cases}\alpha z&z<0\\z&z\geq0\end{cases}",
        "Softplus":r"f(z)=\ln(1+e^z),\quad f'(z)=\frac{1}{1+e^{-z}}"
    }
    st.latex(formulas[act])
    st.caption("For Binary step, the derivative is undefined at zero and zero elsewhere in the ordinary sense; it is not suitable for standard gradient-based learning.")

# 9 Digit recognition
with tabs[8]:
    st.header("Handwritten digit recognition: from pixels to classes")
    st.markdown("""
    A digit-recognition network receives an image as numeric pixel inputs, processes those values
    through hidden layers, and produces one score or probability for each digit class (0–9).
    This tab is a conceptual classroom demo, not a trained digit classifier.
    """)
    st.write("Use the sliders to simulate the output layer's ten class scores.")
    scores=[]
    cols=st.columns(5)
    for i in range(10):
        with cols[i%5]:
            scores.append(st.slider(f"Score for digit {i}",0.0,1.0,float(.1 if i!=7 else .85),.05,key=f"digit{i}"))
    scores=np.array(scores)
    probs=scores/scores.sum() if scores.sum()>0 else np.ones(10)/10
    predicted=int(np.argmax(probs))
    st.metric("Predicted digit",str(predicted))
    fig=go.Figure(go.Bar(x=list(range(10)),y=probs))
    fig.update_layout(height=350,xaxis_title="Digit class",yaxis_title="Normalized score",title="Output layer: relative class probabilities")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("In a real classifier, pixel values are learned through the network; the final layer typically has ten outputs, one for each digit.")

st.divider()
st.caption("Instructor note: numerical examples are designed for live explanation. The XOR weights and core equations follow the supplied lecture notes; additional interactive examples make the ideas easier to explore.")
