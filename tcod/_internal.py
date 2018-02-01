

def verify_order(order):
    order = order.upper()
    if order != 'C' and order != 'F':
        raise TypeError("order must be 'C' or 'F', not %r" % (order,))
    return order

def handle_order(shape, order):
    order = verify_order(order)
    if order == 'C':
        return shape
    else:
        return tuple(reversed(shape))
