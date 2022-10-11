css_style = (
    ".list {\n    font-family:sans-serif;\n  }\n  td {\n    padding:10px;\n    border:solid 1px #eee;"
    "\n    background: #e7e7e7;\n  }\n  thead td{\n    background: #28a8e0;\n    text-align: center;"
    "\n    color: #fff;\n  }\n  input {\n    border:solid 1px #ccc;\n    border-radius: 5px;"
    "\n    padding:7px 14px;\n    margin-bottom:10px\n  }\n  input:focus {\n    outline:none;"
    "\n    border-color:#aaa;\n  }\n  .search {\n    background-color: #eeeeee;\n    font-size: 15px;\n  }"
    "\n  .sort {\n    padding:2px 20px;\n    border-radius: 6px;\n    border:none;\n    display:inline-block;"
    "\n    color:#000;\n    text-decoration: none;\n    background-color: #9eff9c;\n    height:28px;"
    "\n    font-size: 15px;\n  }\n  .sort:hover {\n    text-decoration: none;\n    background-color: #25c766;"
    "\n  }\n  .sort:focus {\n    outline:none;\n  }\n  .sort:after {\n    display:inline-block;\n    width: 0;"
    "\n    height: 0;\n    border-left: 5px solid transparent;\n    border-right: 5px solid transparent;"
    '\n    border-bottom: 5px solid transparent;\n    content:"";\n    position: relative;\n    top:-10px;'
    "\n    right:-5px;\n  }\n  .sort.asc:after {\n    width: 0;\n    height: 0;"
    "\n    border-left: 5px solid transparent;\n    border-right: 5px solid transparent;"
    '\n    border-top: 5px solid #fff;\n    content:"";\n    position: relative;\n    top:4px;'
    "\n    right:-5px;\n  }\n  .sort.desc:after {\n    width: 0;\n    height: 0;"
    "\n    border-left: 5px solid transparent;\n    border-right: 5px solid transparent;"
    '\n    border-bottom: 5px solid #fff;\n    content:"";\n    position: relative;\n    top:-4px;'
    "\n    right:-5px;\n  }\n"
)


class TableVisualizer:
    """
    Render a simple and functional HTML table. Typically used to visualize results.
    table_configs is a list of configs, one element for each column of the tabular data, each of which shoudl contain:
    'id' : Id of the column
    'display_name' : Display name of the column (if different from the 'id')
    'sortable' (bool) : If the column elements require a sort functionality
    'type' (text / image / hyperlink) (str) : This determines the display for the particular column
    'height' : Height on the image (only valid if type is set to image)
    """

    def __init__(self, table_configs, out_file_name):
        self.table_configs = table_configs
        self.validate_configs()
        self.num_columns = len(self.table_configs)
        self.out_file_name = out_file_name
        self.data = []

    def validate_configs(self):
        for column in self.table_configs:
            if "id" not in column:
                raise ValueError(
                    "Field id not present for some elements in the table_configs"
                )
            if "display_name" not in column:
                column["display_name"] = column["id"]
            if "sortable" not in column:
                column["sortable"] = False
            if "type" not in column:
                raise ValueError(
                    "Field type not present for some elements in the table_configs"
                )
            if "height" not in column:
                column["height"] = -1

    def add_row(self, row_data):
        if len(row_data) != self.num_columns:
            raise ValueError("Incorrect number of entries in the row data.")
        self.data.append(row_data)

    def render(self):
        with open(self.out_file_name, "w") as f:
            f.write("<html>\n")

            f.write('<div id="viztable">\n')
            self.add_search_sort_functionality(f)

            f.write("<table>\n")
            self.add_table_head(f)
            self.add_table_body(f)
            f.write("</table>\n")

            f.write("</div>\n")
            self.add_html_head(f)
            f.write("</html>")

    def add_search_sort_functionality(self, f):
        f.write('<input class="search" placeholder="Search" />\n<br>\n')
        for column in self.table_configs:
            if column["sortable"]:
                f.write('<button class="sort" data-sort="{0}">'.format(column["id"]))
                f.write("Sort by {0}".format(column["display_name"]))
                f.write("</button>\n")
        f.write("<p>\n")

    def add_table_head(self, f):
        f.write("<thead>\n")
        f.write("<tr>\n")
        for column in self.table_configs:
            f.write(
                '<td class="{0}">{1}</td>\n'.format(
                    column["id"], column["display_name"]
                )
            )
        f.write("</tr>\n")
        f.write("</thead>\n")

    def add_table_body(self, f):
        f.write('<tbody class="list">\n')

        for current_data in self.data:
            f.write("<tr>\n")
            for col_count, col_data in enumerate(current_data):
                f.write('<td class="{}">\n'.format(self.table_configs[col_count]["id"]))
                if self.table_configs[col_count]["type"] == "text":
                    f.write("{}".format(col_data))
                elif self.table_configs[col_count]["type"] == "image":
                    f.write(
                        '<img src="{}" height={}>'.format(
                            col_data, self.table_configs[col_count]["height"]
                        )
                    )
                elif self.table_configs[col_count]["type"] == "video":
                    f.write(
                        '<video height={} controls src="{}" type="video/mp4">'.format(
                            self.table_configs[col_count]["height"], col_data
                        )
                    )
                elif self.table_configs[col_count]["type"] == "hyperlink":
                    f.write(
                        '<a href="{}" target=“_blank”>{}</a>'.format(
                            col_data[0], col_data[1]
                        )
                    )
                f.write("</td>\n")
            f.write("</tr>\n")

        f.write("</tbody>\n")

    def add_html_head(self, f):
        f.write("<head>\n")
        f.write("<title>Data visualization</title>\n")
        f.write(
            '<script src="http://cdnjs.cloudflare.com/ajax/libs/list.js/1.5.0/list.min.js"></script>\n'
        )
        self.add_js(f)
        self.add_style(f)
        f.write("</head>")

    def add_js(self, f):
        column_names_repr = []
        for column in self.table_configs:
            column_names_repr.append('"{}"'.format(column["id"]))
        column_names_repr = ",".join(column_names_repr)

        f.write("<script>\n")
        f.write("var options = {{valueNames: [ {0} ]}};\n".format(column_names_repr))
        f.write('var gameList = new List("viztable", options);\n')
        f.write("</script>\n")

    def add_style(self, f):
        f.write("<style>\n")
        f.writelines(css_style)
        f.write("</style>\n")
